"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from "recharts";
import {
  History,
  TrendingUp,
  TrendingDown,
  Target,
  BarChart3,
  Calendar,
} from "lucide-react";

interface Analog {
  period: {
    start: string;
    end: string;
  };
  similarity: number;
  conditions_at_match: {
    price: number;
    volatility: number;
    trend_20d: number;
    rsi: number;
  };
  forward_returns: {
    "7d": number;
    "14d": number;
    "30d": number;
    "60d": number;
  };
  risk_profile: {
    max_drawdown_30d: number;
    max_gain_30d: number;
  };
  context: {
    event: string;
    regime: string;
  };
}

interface AnalogData {
  asset: string;
  analysis_date: string;
  current_conditions: {
    price: number;
    volatility: number;
    trend_20d: number;
    rsi: number;
    regime: string;
  };
  analogs: Analog[];
  statistics: {
    count: number;
    avg_return_7d: number;
    avg_return_14d: number;
    avg_return_30d: number;
    avg_return_60d: number;
    win_rate_30d: number;
    median_return_30d: number;
    best_case_30d: number;
    worst_case_30d: number;
  };
  forecast: {
    direction: string;
    confidence: number;
    narrative: string;
  };
}

interface Props {
  data?: AnalogData;
  assetId?: string;
}

export function HistoricalAnalogChart({ data, assetId }: Props) {
  const [analogData, setAnalogData] = useState<AnalogData | null>(data || null);
  const [selectedAnalog, setSelectedAnalog] = useState<number | null>(null);

  useEffect(() => {
    if (!data && assetId) {
      fetch(`/api/v2/analogs/${assetId}`)
        .then((res) => res.json())
        .then((d) => setAnalogData(d))
        .catch(console.error);
    }
  }, [assetId, data]);

  // Generate projection chart data
  const chartData = useMemo(() => {
    if (!analogData) return [];

    const days = [0, 7, 14, 30, 60];
    
    // Build data points for each day
    return days.map((day) => {
      const point: Record<string, number | string> = { day: `D+${day}` };
      
      // Add current price baseline (normalized to 100)
      point.current = 100;
      
      // Add each analog's projected path
      analogData.analogs.forEach((analog, i) => {
        const returnVal = 
          day === 0 ? 0 :
          day === 7 ? analog.forward_returns["7d"] :
          day === 14 ? analog.forward_returns["14d"] :
          day === 30 ? analog.forward_returns["30d"] :
          analog.forward_returns["60d"];
        
        point[`analog${i}`] = 100 + returnVal;
      });
      
      // Add average line
      const avgReturn = 
        day === 0 ? 0 :
        day === 7 ? analogData.statistics.avg_return_7d :
        day === 14 ? analogData.statistics.avg_return_14d :
        day === 30 ? analogData.statistics.avg_return_30d :
        analogData.statistics.avg_return_60d;
      
      point.average = 100 + avgReturn;
      
      // Add best/worst case for band
      point.best = 100 + (day === 30 ? analogData.statistics.best_case_30d : avgReturn * 1.5);
      point.worst = 100 + (day === 30 ? analogData.statistics.worst_case_30d : avgReturn * 0.5 - 2);
      
      return point;
    });
  }, [analogData]);

  if (!analogData) {
    return (
      <Card className="col-span-2">
        <CardContent className="flex items-center justify-center h-64">
          <div className="animate-pulse text-muted-foreground">Loading historical analogs...</div>
        </CardContent>
      </Card>
    );
  }

  const forecastColors: Record<string, string> = {
    BULLISH: "text-green-500",
    BEARISH: "text-red-500",
    NEUTRAL: "text-yellow-500",
  };

  const analogColors = [
    "#3b82f6", "#8b5cf6", "#ec4899", "#f97316", "#84cc16",
    "#06b6d4", "#f59e0b", "#10b981", "#6366f1", "#ef4444"
  ];

  return (
    <div className="space-y-4">
      {/* Main Chart */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg flex items-center gap-2">
                <History className="h-5 w-5" />
                Historical Analog Projections
              </CardTitle>
              <CardDescription>
                {analogData.statistics.count} similar setups found — projecting forward 60 days
              </CardDescription>
            </div>
            <Badge 
              variant="outline"
              className={`${forecastColors[analogData.forecast.direction]} border-current`}
            >
              {analogData.forecast.direction} ({analogData.forecast.confidence.toFixed(0)}%)
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis 
                  dataKey="day" 
                  tick={{ fontSize: 12 }}
                />
                <YAxis 
                  domain={["dataMin - 5", "dataMax + 5"]}
                  tickFormatter={(v) => `${v.toFixed(0)}%`}
                  tick={{ fontSize: 12 }}
                />
                <Tooltip 
                  formatter={(value: number, name: string) => {
                    if (name === "average") return [`${value.toFixed(1)}%`, "Average"];
                    if (name === "current") return [`${value.toFixed(1)}%`, "Current"];
                    if (name.startsWith("analog")) {
                      const idx = parseInt(name.replace("analog", ""));
                      const analog = analogData.analogs[idx];
                      return [`${value.toFixed(1)}%`, analog?.period.start || name];
                    }
                    return [`${value.toFixed(1)}%`, name];
                  }}
                  contentStyle={{
                    backgroundColor: "hsl(var(--background))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
                
                {/* Projection band (best to worst case) */}
                <Area
                  type="monotone"
                  dataKey="best"
                  stroke="none"
                  fill="#22c55e"
                  fillOpacity={0.1}
                />
                <Area
                  type="monotone"
                  dataKey="worst"
                  stroke="none"
                  fill="#ef4444"
                  fillOpacity={0.1}
                />
                
                {/* Individual analog lines */}
                {analogData.analogs.slice(0, 5).map((analog, i) => (
                  <Line
                    key={`analog${i}`}
                    type="monotone"
                    dataKey={`analog${i}`}
                    stroke={analogColors[i]}
                    strokeWidth={selectedAnalog === i ? 3 : 1.5}
                    strokeOpacity={selectedAnalog === null || selectedAnalog === i ? 0.7 : 0.2}
                    dot={false}
                    strokeDasharray="5 5"
                  />
                ))}
                
                {/* Average line */}
                <Line
                  type="monotone"
                  dataKey="average"
                  stroke="#ffffff"
                  strokeWidth={3}
                  dot={{ fill: "#ffffff", r: 4 }}
                />
                
                {/* Reference line at 100 (current) */}
                <ReferenceLine 
                  y={100} 
                  stroke="hsl(var(--muted-foreground))" 
                  strokeDasharray="3 3"
                  label={{ value: "Current", position: "left", fontSize: 10 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          
          {/* Legend */}
          <div className="mt-4 flex flex-wrap gap-2">
            <div className="flex items-center gap-1.5 text-xs">
              <div className="w-8 h-0.5 bg-white rounded" />
              <span>Average Path</span>
            </div>
            {analogData.analogs.slice(0, 5).map((analog, i) => (
              <button
                key={i}
                className={`flex items-center gap-1.5 text-xs px-2 py-1 rounded transition-all ${
                  selectedAnalog === i ? "bg-muted" : "hover:bg-muted/50"
                }`}
                onClick={() => setSelectedAnalog(selectedAnalog === i ? null : i)}
              >
                <div 
                  className="w-4 h-0.5 rounded" 
                  style={{ backgroundColor: analogColors[i] }}
                />
                <span>{analog.period.start}</span>
                <span className="text-muted-foreground">({analog.similarity}%)</span>
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Statistics Cards */}
      <div className="grid grid-cols-4 gap-4">
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <Target className="h-4 w-4" />
              <span className="text-xs">30-Day Win Rate</span>
            </div>
            <div className={`text-2xl font-bold ${
              analogData.statistics.win_rate_30d > 60 ? "text-green-500" : 
              analogData.statistics.win_rate_30d < 40 ? "text-red-500" : "text-yellow-500"
            }`}>
              {analogData.statistics.win_rate_30d.toFixed(0)}%
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <BarChart3 className="h-4 w-4" />
              <span className="text-xs">Avg 30-Day Return</span>
            </div>
            <div className={`text-2xl font-bold ${
              analogData.statistics.avg_return_30d > 0 ? "text-green-500" : "text-red-500"
            }`}>
              {analogData.statistics.avg_return_30d > 0 ? "+" : ""}
              {analogData.statistics.avg_return_30d.toFixed(1)}%
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <TrendingUp className="h-4 w-4 text-green-500" />
              <span className="text-xs">Best Case</span>
            </div>
            <div className="text-2xl font-bold text-green-500">
              +{analogData.statistics.best_case_30d.toFixed(1)}%
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="pt-4">
            <div className="flex items-center gap-2 text-muted-foreground mb-1">
              <TrendingDown className="h-4 w-4 text-red-500" />
              <span className="text-xs">Worst Case</span>
            </div>
            <div className="text-2xl font-bold text-red-500">
              {analogData.statistics.worst_case_30d.toFixed(1)}%
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Forecast Narrative */}
      <Card className="bg-gradient-to-r from-purple-500/10 to-blue-500/10">
        <CardContent className="pt-4">
          <p className="text-sm leading-relaxed">{analogData.forecast.narrative}</p>
        </CardContent>
      </Card>

      {/* Detailed Analog Table */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Calendar className="h-4 w-4" />
            Historical Analog Details
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-2 px-2">Period</th>
                  <th className="text-right py-2 px-2">Match</th>
                  <th className="text-right py-2 px-2">Vol</th>
                  <th className="text-right py-2 px-2">Trend</th>
                  <th className="text-right py-2 px-2">RSI</th>
                  <th className="text-right py-2 px-2">+7d</th>
                  <th className="text-right py-2 px-2">+14d</th>
                  <th className="text-right py-2 px-2">+30d</th>
                  <th className="text-right py-2 px-2">+60d</th>
                  <th className="text-left py-2 px-2">Context</th>
                </tr>
              </thead>
              <tbody>
                {analogData.analogs.map((analog, i) => (
                  <tr 
                    key={i}
                    className={`border-b hover:bg-muted/50 cursor-pointer ${
                      selectedAnalog === i ? "bg-muted" : ""
                    }`}
                    onClick={() => setSelectedAnalog(selectedAnalog === i ? null : i)}
                  >
                    <td className="py-2 px-2">
                      <div className="flex items-center gap-2">
                        <div 
                          className="w-2 h-2 rounded-full" 
                          style={{ backgroundColor: analogColors[i] }}
                        />
                        <span>{analog.period.start}</span>
                      </div>
                    </td>
                    <td className="text-right py-2 px-2 font-medium">{analog.similarity}%</td>
                    <td className="text-right py-2 px-2">{analog.conditions_at_match.volatility.toFixed(0)}%</td>
                    <td className={`text-right py-2 px-2 ${
                      analog.conditions_at_match.trend_20d > 0 ? "text-green-500" : "text-red-500"
                    }`}>
                      {analog.conditions_at_match.trend_20d > 0 ? "+" : ""}
                      {analog.conditions_at_match.trend_20d.toFixed(1)}%
                    </td>
                    <td className="text-right py-2 px-2">{analog.conditions_at_match.rsi.toFixed(0)}</td>
                    <td className={`text-right py-2 px-2 ${
                      analog.forward_returns["7d"] > 0 ? "text-green-500" : "text-red-500"
                    }`}>
                      {analog.forward_returns["7d"] > 0 ? "+" : ""}
                      {analog.forward_returns["7d"].toFixed(1)}%
                    </td>
                    <td className={`text-right py-2 px-2 ${
                      analog.forward_returns["14d"] > 0 ? "text-green-500" : "text-red-500"
                    }`}>
                      {analog.forward_returns["14d"] > 0 ? "+" : ""}
                      {analog.forward_returns["14d"].toFixed(1)}%
                    </td>
                    <td className={`text-right py-2 px-2 font-medium ${
                      analog.forward_returns["30d"] > 0 ? "text-green-500" : "text-red-500"
                    }`}>
                      {analog.forward_returns["30d"] > 0 ? "+" : ""}
                      {analog.forward_returns["30d"].toFixed(1)}%
                    </td>
                    <td className={`text-right py-2 px-2 ${
                      analog.forward_returns["60d"] > 0 ? "text-green-500" : "text-red-500"
                    }`}>
                      {analog.forward_returns["60d"] > 0 ? "+" : ""}
                      {analog.forward_returns["60d"].toFixed(1)}%
                    </td>
                    <td className="py-2 px-2 text-muted-foreground">
                      {analog.context.event || analog.context.regime}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Current Conditions */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Current Conditions Being Matched</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-5 gap-4">
            <div>
              <div className="text-xs text-muted-foreground">Price</div>
              <div className="font-medium">${analogData.current_conditions.price.toLocaleString()}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Volatility</div>
              <div className="font-medium">{analogData.current_conditions.volatility.toFixed(1)}%</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">20-Day Trend</div>
              <div className={`font-medium ${
                analogData.current_conditions.trend_20d > 0 ? "text-green-500" : "text-red-500"
              }`}>
                {analogData.current_conditions.trend_20d > 0 ? "+" : ""}
                {analogData.current_conditions.trend_20d.toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">RSI</div>
              <div className="font-medium">{analogData.current_conditions.rsi.toFixed(0)}</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Regime</div>
              <div className="font-medium">{analogData.current_conditions.regime}</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export default HistoricalAnalogChart;
