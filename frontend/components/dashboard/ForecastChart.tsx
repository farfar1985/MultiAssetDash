"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
  ReferenceLine,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  Target,
  Calendar,
  DollarSign,
  Percent,
  Activity,
  Info,
} from "lucide-react";

interface ForecastData {
  asset: string;
  currentPrice: number;
  forecast: {
    days: number;
    targetPrice: number;
    percentChange: number;
    confidence: number;
    lowTarget: number;
    highTarget: number;
  }[];
  signals: {
    direction: "BULLISH" | "BEARISH" | "NEUTRAL";
    strength: number;
    winRate: number;
  };
  historicalPath: {
    date: string;
    price: number;
  }[];
}

interface Props {
  assetId: string;
  data?: ForecastData;
}

// Generate mock forecast data for demo
const generateForecastData = (assetId: string): ForecastData => {
  const basePrices: Record<string, number> = {
    SP500: 5892.34,
    NASDAQ: 18234.56,
    GOLD: 2847.20,
    CRUDE: 72.45,
    BITCOIN: 98234.00,
  };

  const basePrice = basePrices[assetId] || 100;
  const volatility = assetId === "BITCOIN" ? 0.15 : assetId === "CRUDE" ? 0.08 : 0.04;
  const trend = assetId === "GOLD" ? 0.03 : assetId === "CRUDE" ? -0.02 : 0.05;

  // Generate historical path (last 30 days)
  const historicalPath = [];
  let price = basePrice * (1 - trend * 0.5); // Start 30 days ago
  for (let i = 30; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    price = price * (1 + (Math.random() - 0.5) * volatility * 0.1 + trend / 30);
    historicalPath.push({
      date: date.toISOString().split("T")[0],
      price: price,
    });
  }

  // Generate forecast
  const forecast = [7, 14, 30, 60].map((days) => {
    const expectedChange = trend * (days / 30);
    const uncertainty = volatility * Math.sqrt(days / 30);
    
    return {
      days,
      targetPrice: basePrice * (1 + expectedChange),
      percentChange: expectedChange * 100,
      confidence: Math.max(40, 85 - days * 0.5),
      lowTarget: basePrice * (1 + expectedChange - uncertainty),
      highTarget: basePrice * (1 + expectedChange + uncertainty),
    };
  });

  return {
    asset: assetId,
    currentPrice: basePrice,
    forecast,
    signals: {
      direction: trend > 0.02 ? "BULLISH" : trend < -0.02 ? "BEARISH" : "NEUTRAL",
      strength: Math.min(95, Math.abs(trend) * 1000),
      winRate: 65 + Math.random() * 15,
    },
    historicalPath,
  };
};

export function ForecastChart({ assetId, data }: Props) {
  const [forecastData, setForecastData] = useState<ForecastData | null>(data || null);
  const [selectedHorizon, setSelectedHorizon] = useState<number>(30);
  const [showConfidenceBand, setShowConfidenceBand] = useState(true);

  useEffect(() => {
    if (!data) {
      // In production, this would fetch from API
      setForecastData(generateForecastData(assetId));
    }
  }, [assetId, data]);

  // Build chart data combining historical + forecast
  const chartData = useMemo(() => {
    if (!forecastData) return [];

    const result: Record<string, number | string | undefined>[] = [];
    
    // Add historical data
    forecastData.historicalPath.forEach((point) => {
      result.push({
        date: point.date,
        price: point.price,
        type: "historical",
      });
    });

    // Add forecast points
    const today = new Date();
    
    forecastData.forecast.forEach((f) => {
      const futureDate = new Date(today);
      futureDate.setDate(futureDate.getDate() + f.days);
      
      result.push({
        date: futureDate.toISOString().split("T")[0],
        forecast: f.targetPrice,
        forecastHigh: f.highTarget,
        forecastLow: f.lowTarget,
        type: "forecast",
      });
    });

    return result;
  }, [forecastData]);

  const selectedForecast = forecastData?.forecast.find((f) => f.days === selectedHorizon);

  if (!forecastData) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-96">
          <Activity className="h-6 w-6 animate-spin mr-2" />
          <span>Loading forecast...</span>
        </CardContent>
      </Card>
    );
  }

  const formatPrice = (value: number) => {
    if (value >= 10000) return `$${(value / 1000).toFixed(1)}K`;
    if (value >= 1000) return `$${value.toFixed(0)}`;
    return `$${value.toFixed(2)}`;
  };

  const isBullish = forecastData.signals.direction === "BULLISH";
  const isBearish = forecastData.signals.direction === "BEARISH";

  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl flex items-center gap-2">
              <Target className="h-5 w-5" />
              Price Forecast
            </CardTitle>
            <CardDescription>
              AI-powered price projection based on signal confluence
            </CardDescription>
          </div>
          <Badge
            className={`text-white ${
              isBullish ? "bg-green-500" : isBearish ? "bg-red-500" : "bg-yellow-500"
            }`}
          >
            {forecastData.signals.direction}
          </Badge>
        </div>
      </CardHeader>

      <CardContent>
        {/* Horizon Selector */}
        <div className="flex items-center gap-2 mb-4">
          <span className="text-sm text-muted-foreground">Horizon:</span>
          {[7, 14, 30, 60].map((days) => (
            <Button
              key={days}
              variant={selectedHorizon === days ? "default" : "outline"}
              size="sm"
              onClick={() => setSelectedHorizon(days)}
            >
              {days}D
            </Button>
          ))}
          <div className="flex-1" />
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowConfidenceBand(!showConfidenceBand)}
          >
            <Info className="h-4 w-4 mr-1" />
            {showConfidenceBand ? "Hide" : "Show"} Range
          </Button>
        </div>

        {/* Chart */}
        <div className="h-80 mb-4">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData}>
              <defs>
                <linearGradient id="forecastGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={isBullish ? "#22c55e" : "#ef4444"} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={isBullish ? "#22c55e" : "#ef4444"} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="confidenceBand" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              
              <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
              <XAxis 
                dataKey="date" 
                tick={{ fontSize: 10 }}
                tickFormatter={(d) => {
                  const date = new Date(d);
                  return `${date.getMonth() + 1}/${date.getDate()}`;
                }}
              />
              <YAxis 
                domain={["dataMin * 0.95", "dataMax * 1.05"]}
                tick={{ fontSize: 10 }}
                tickFormatter={formatPrice}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--background))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                }}
                formatter={(value: number, name: string) => {
                  const label = name === "price" ? "Actual" : 
                               name === "forecast" ? "Forecast" :
                               name === "forecastHigh" ? "High Target" :
                               name === "forecastLow" ? "Low Target" : name;
                  return [formatPrice(value), label];
                }}
              />

              {/* Confidence band */}
              {showConfidenceBand && (
                <>
                  <Area
                    type="monotone"
                    dataKey="forecastHigh"
                    stroke="none"
                    fill="url(#confidenceBand)"
                    connectNulls
                  />
                  <Area
                    type="monotone"
                    dataKey="forecastLow"
                    stroke="none"
                    fill="none"
                    connectNulls
                  />
                </>
              )}

              {/* Historical price line */}
              <Line
                type="monotone"
                dataKey="price"
                stroke="#ffffff"
                strokeWidth={2}
                dot={false}
                connectNulls
              />

              {/* Forecast line */}
              <Line
                type="monotone"
                dataKey="forecast"
                stroke={isBullish ? "#22c55e" : isBearish ? "#ef4444" : "#eab308"}
                strokeWidth={3}
                strokeDasharray="8 4"
                dot={{ fill: isBullish ? "#22c55e" : isBearish ? "#ef4444" : "#eab308", r: 5 }}
                connectNulls
              />

              {/* Today reference line */}
              <ReferenceLine
                x={new Date().toISOString().split("T")[0]}
                stroke="hsl(var(--muted-foreground))"
                strokeDasharray="3 3"
                label={{ value: "Today", position: "top", fontSize: 10 }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* Forecast Details */}
        {selectedForecast && (
          <div className="grid grid-cols-4 gap-4 p-4 bg-muted/30 rounded-lg">
            <div>
              <div className="flex items-center gap-1 text-muted-foreground mb-1">
                <Calendar className="h-4 w-4" />
                <span className="text-xs">Horizon</span>
              </div>
              <div className="text-xl font-bold">{selectedForecast.days} Days</div>
            </div>
            
            <div>
              <div className="flex items-center gap-1 text-muted-foreground mb-1">
                <DollarSign className="h-4 w-4" />
                <span className="text-xs">Target Price</span>
              </div>
              <div className="text-xl font-bold">{formatPrice(selectedForecast.targetPrice)}</div>
              <div className="text-xs text-muted-foreground">
                Range: {formatPrice(selectedForecast.lowTarget)} - {formatPrice(selectedForecast.highTarget)}
              </div>
            </div>
            
            <div>
              <div className="flex items-center gap-1 text-muted-foreground mb-1">
                <Percent className="h-4 w-4" />
                <span className="text-xs">Expected Return</span>
              </div>
              <div className={`text-xl font-bold ${
                selectedForecast.percentChange > 0 ? "text-green-500" : "text-red-500"
              }`}>
                {selectedForecast.percentChange > 0 ? "+" : ""}
                {selectedForecast.percentChange.toFixed(1)}%
              </div>
            </div>
            
            <div>
              <div className="flex items-center gap-1 text-muted-foreground mb-1">
                <Target className="h-4 w-4" />
                <span className="text-xs">Confidence</span>
              </div>
              <div className="text-xl font-bold">{selectedForecast.confidence.toFixed(0)}%</div>
              <div className="h-2 bg-muted rounded-full overflow-hidden mt-1">
                <div 
                  className={`h-full ${
                    selectedForecast.confidence > 70 ? "bg-green-500" :
                    selectedForecast.confidence > 50 ? "bg-yellow-500" : "bg-red-500"
                  }`}
                  style={{ width: `${selectedForecast.confidence}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Signal Summary */}
        <div className="grid grid-cols-3 gap-4 mt-4">
          <Card className={`${isBullish ? "bg-green-500/10 border-green-500/30" : isBearish ? "bg-red-500/10 border-red-500/30" : "bg-yellow-500/10 border-yellow-500/30"}`}>
            <CardContent className="pt-3">
              <div className="flex items-center gap-2">
                {isBullish ? (
                  <TrendingUp className="h-5 w-5 text-green-500" />
                ) : isBearish ? (
                  <TrendingDown className="h-5 w-5 text-red-500" />
                ) : (
                  <Activity className="h-5 w-5 text-yellow-500" />
                )}
                <div>
                  <div className="text-xs text-muted-foreground">Signal Direction</div>
                  <div className="font-bold">{forecastData.signals.direction}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-3">
              <div className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                <div>
                  <div className="text-xs text-muted-foreground">Signal Strength</div>
                  <div className="font-bold">{forecastData.signals.strength.toFixed(0)}%</div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-3">
              <div className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                <div>
                  <div className="text-xs text-muted-foreground">Historical Win Rate</div>
                  <div className="font-bold">{forecastData.signals.winRate.toFixed(0)}%</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Disclaimer */}
        <p className="text-xs text-muted-foreground mt-4 text-center">
          Forecasts are based on historical patterns and signal confluence. Past performance does not guarantee future results.
        </p>
      </CardContent>
    </Card>
  );
}

export default ForecastChart;
