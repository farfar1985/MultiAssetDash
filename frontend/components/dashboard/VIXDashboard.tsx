"use client";

import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Shield,
  BarChart3,
} from "lucide-react";

interface VIXData {
  success: boolean;
  timestamp: string;
  vix: {
    level: number;
    percentile: number;
    level_signal: string;
    change_5d: number;
    change_20d: number;
  };
  signals: {
    spike: string;
    cot: string;
    overall: string;
  };
  cot_positioning: {
    commercial_net: number;
    dealer_net: number;
    leveraged_net: number;
  };
  futures: {
    open_interest: number;
    volume: number;
  };
  recommendation: {
    signal: string;
    confidence: number;
    message: string;
    reasoning: string[];
  };
}

const SignalBadge = ({ signal }: { signal: string }) => {
  const colors: Record<string, string> = {
    BUY_EQUITIES: "bg-green-500",
    SELL_EQUITIES: "bg-red-500",
    NEUTRAL: "bg-gray-400",
    FEAR_EXTREME: "bg-green-600",
    FEAR_HIGH: "bg-green-500",
    ELEVATED: "bg-yellow-500",
    NORMAL: "bg-blue-400",
    COMPLACENT: "bg-orange-400",
    EXTREME_COMPLACENCY: "bg-red-500",
  };

  return (
    <Badge className={`${colors[signal] || "bg-gray-400"} text-white`}>
      {signal.replace(/_/g, " ")}
    </Badge>
  );
};

const VIXGauge = ({ value, percentile }: { value: number; percentile: number }) => {
  // VIX color based on level
  const getColor = () => {
    if (value >= 40) return "text-green-500"; // Fear extreme = buy signal
    if (value >= 30) return "text-yellow-500";
    if (value >= 25) return "text-orange-500";
    if (value >= 15) return "text-blue-500";
    if (value >= 12) return "text-orange-400";
    return "text-red-500"; // Extreme complacency
  };

  return (
    <div className="text-center">
      <div className={`text-5xl font-bold ${getColor()}`}>
        {value.toFixed(1)}
      </div>
      <div className="text-sm text-muted-foreground mt-1">
        {percentile.toFixed(0)}th percentile
      </div>
      <Progress value={percentile} className="h-2 mt-2" />
    </div>
  );
};

export function VIXDashboard() {
  const [data, setData] = useState<VIXData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("/api/v2/vix");
        const json = await res.json();
        
        if (json.success) {
          setData(json);
        } else {
          setError(json.error || "Failed to load VIX data");
        }
        setLoading(false);
      } catch (_e) {
        setError("Failed to fetch VIX data");
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 60000);
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

  if (error || !data) {
    return (
      <Card>
        <CardContent className="p-6 text-red-500">
          {error || "No VIX data available"}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          VIX Intelligence
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* VIX Level */}
          <div className="flex flex-col items-center p-4 bg-muted/50 rounded-lg">
            <VIXGauge 
              value={data.vix.level} 
              percentile={data.vix.percentile} 
            />
            <div className="mt-3">
              <SignalBadge signal={data.vix.level_signal} />
            </div>
          </div>

          {/* Changes & Signals */}
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-2">
              <div className="p-3 bg-muted/30 rounded">
                <div className="text-xs text-muted-foreground">5-Day</div>
                <div className={`text-lg font-medium flex items-center gap-1 ${
                  data.vix.change_5d > 0 ? "text-red-500" : "text-green-500"
                }`}>
                  {data.vix.change_5d > 0 ? (
                    <TrendingUp className="h-4 w-4" />
                  ) : (
                    <TrendingDown className="h-4 w-4" />
                  )}
                  {data.vix.change_5d > 0 ? "+" : ""}
                  {data.vix.change_5d.toFixed(1)}%
                </div>
              </div>
              <div className="p-3 bg-muted/30 rounded">
                <div className="text-xs text-muted-foreground">20-Day</div>
                <div className={`text-lg font-medium flex items-center gap-1 ${
                  data.vix.change_20d > 0 ? "text-red-500" : "text-green-500"
                }`}>
                  {data.vix.change_20d > 0 ? (
                    <TrendingUp className="h-4 w-4" />
                  ) : (
                    <TrendingDown className="h-4 w-4" />
                  )}
                  {data.vix.change_20d > 0 ? "+" : ""}
                  {data.vix.change_20d.toFixed(1)}%
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Spike Signal</span>
                <Badge variant={data.signals.spike === "SPIKE" ? "destructive" : "secondary"}>
                  {data.signals.spike}
                </Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">COT Signal</span>
                <Badge variant={
                  data.signals.cot === "BULLISH" ? "default" : 
                  data.signals.cot === "BEARISH" ? "destructive" : "secondary"
                }>
                  {data.signals.cot}
                </Badge>
              </div>
            </div>
          </div>

          {/* COT Positioning */}
          <div className="space-y-3">
            <div className="text-sm font-medium flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              COT Positioning
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Commercial</span>
                <span className={data.cot_positioning.commercial_net > 0 ? "text-green-500" : "text-red-500"}>
                  {data.cot_positioning.commercial_net.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Dealer</span>
                <span className={data.cot_positioning.dealer_net > 0 ? "text-green-500" : "text-red-500"}>
                  {data.cot_positioning.dealer_net.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Leveraged</span>
                <span className={data.cot_positioning.leveraged_net > 0 ? "text-green-500" : "text-red-500"}>
                  {data.cot_positioning.leveraged_net.toLocaleString()}
                </span>
              </div>
            </div>
            <div className="pt-2 border-t text-xs text-muted-foreground">
              OI: {data.futures.open_interest.toLocaleString()} | Vol: {data.futures.volume.toLocaleString()}
            </div>
          </div>
        </div>

        {/* Recommendation */}
        <div className={`mt-4 p-4 rounded-lg ${
          data.recommendation.signal === "BUY_EQUITIES" 
            ? "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800"
            : data.recommendation.signal === "SELL_EQUITIES"
            ? "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800"
            : "bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800"
        }`}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              <span className="font-medium">Recommendation</span>
            </div>
            <div className="flex items-center gap-2">
              <SignalBadge signal={data.recommendation.signal} />
              <span className="text-sm text-muted-foreground">
                {data.recommendation.confidence}% confidence
              </span>
            </div>
          </div>
          <p className="text-sm">{data.recommendation.message}</p>
          
          {data.recommendation.reasoning.length > 0 && (
            <div className="mt-2 space-y-1">
              {data.recommendation.reasoning.map((r, i) => (
                <div key={i} className="text-xs text-muted-foreground flex items-start gap-1">
                  <AlertTriangle className="h-3 w-3 mt-0.5 flex-shrink-0" />
                  {r}
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default VIXDashboard;
