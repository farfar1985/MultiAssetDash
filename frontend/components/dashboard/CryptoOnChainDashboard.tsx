"use client";

import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Bitcoin,
  TrendingUp,
  TrendingDown,
  Activity,
  Cpu,
  Wallet,
} from "lucide-react";

interface CryptoData {
  success: boolean;
  timestamp: string;
  asset: string;
  signals: {
    mvrv: string;
    nvt: string;
    activity: string;
    overall: string;
  };
  metrics: {
    mvrv: number;
    mvrv_percentile: number;
    nvt: number;
    nvt_percentile: number;
    active_addresses: number;
    hashrate: number;
    transfer_value_usd: number;
    addr_30d_change: number;
    hashrate_30d_change: number;
  };
  recommendation: {
    signal: string;
    confidence: number;
    message: string;
    reasoning: string[];
    is_undervalued: boolean;
    is_overvalued: boolean;
  };
}

const SignalBadge = ({ signal }: { signal: string }) => {
  const colors: Record<string, string> = {
    STRONG_BUY: "bg-green-600",
    BUY: "bg-green-500",
    ACCUMULATE: "bg-green-400",
    NEUTRAL: "bg-gray-400",
    DISTRIBUTE: "bg-orange-400",
    SELL: "bg-red-500",
    STRONG_SELL: "bg-red-600",
  };

  return (
    <Badge className={`${colors[signal] || "bg-gray-400"} text-white`}>
      {signal.replace(/_/g, " ")}
    </Badge>
  );
};

const MVRVGauge = ({ value, percentile }: { value: number; percentile: number }) => {
  // MVRV zones: <1 = buy, 1-2.5 = fair, >3 = sell
  const getColor = () => {
    if (value < 1.0) return "text-green-600";
    if (value < 1.5) return "text-green-500";
    if (value < 2.5) return "text-blue-500";
    if (value < 3.0) return "text-orange-500";
    return "text-red-500";
  };

  const getZone = () => {
    if (value < 1.0) return "Deep Value";
    if (value < 1.5) return "Undervalued";
    if (value < 2.5) return "Fair Value";
    if (value < 3.0) return "Overvalued";
    return "Bubble";
  };

  return (
    <div className="text-center p-4 bg-muted/50 rounded-lg">
      <div className="text-sm text-muted-foreground mb-1">MVRV Ratio</div>
      <div className={`text-4xl font-bold ${getColor()}`}>
        {value.toFixed(2)}
      </div>
      <div className="text-sm font-medium mt-1">{getZone()}</div>
      <div className="text-xs text-muted-foreground mt-1">
        {percentile.toFixed(0)}th percentile
      </div>
    </div>
  );
};

const formatNumber = (n: number): string => {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(0);
};

export function CryptoOnChainDashboard() {
  const [data, setData] = useState<CryptoData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch("/api/v2/crypto");
        const json = await res.json();

        if (json.success) {
          setData(json);
        } else {
          setError(json.error || "Failed to load crypto data");
        }
        setLoading(false);
      } catch {
        setError("Failed to fetch crypto data");
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
          {error || "No crypto data available"}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2">
          <Bitcoin className="h-5 w-5 text-orange-500" />
          Bitcoin On-Chain Intelligence
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
          {/* MVRV Gauge */}
          <MVRVGauge
            value={data.metrics.mvrv}
            percentile={data.metrics.mvrv_percentile}
          />

          {/* NVT */}
          <div className="text-center p-4 bg-muted/50 rounded-lg">
            <div className="text-sm text-muted-foreground mb-1">NVT Ratio</div>
            <div className={`text-3xl font-bold ${
              data.metrics.nvt < 50 ? "text-green-500" :
              data.metrics.nvt > 150 ? "text-red-500" : "text-blue-500"
            }`}>
              {data.metrics.nvt.toFixed(0)}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {data.metrics.nvt_percentile.toFixed(0)}th percentile
            </div>
            <SignalBadge signal={data.signals.nvt} />
          </div>

          {/* Active Addresses */}
          <div className="text-center p-4 bg-muted/50 rounded-lg">
            <div className="text-sm text-muted-foreground mb-1 flex items-center justify-center gap-1">
              <Wallet className="h-3 w-3" />
              Active Addresses
            </div>
            <div className="text-2xl font-bold">
              {formatNumber(data.metrics.active_addresses)}
            </div>
            <div className={`text-sm flex items-center justify-center gap-1 ${
              data.metrics.addr_30d_change > 0 ? "text-green-500" : "text-red-500"
            }`}>
              {data.metrics.addr_30d_change > 0 ? (
                <TrendingUp className="h-3 w-3" />
              ) : (
                <TrendingDown className="h-3 w-3" />
              )}
              {data.metrics.addr_30d_change > 0 ? "+" : ""}
              {data.metrics.addr_30d_change.toFixed(1)}% (30d)
            </div>
          </div>

          {/* Hashrate */}
          <div className="text-center p-4 bg-muted/50 rounded-lg">
            <div className="text-sm text-muted-foreground mb-1 flex items-center justify-center gap-1">
              <Cpu className="h-3 w-3" />
              Hashrate
            </div>
            <div className="text-2xl font-bold">
              {(data.metrics.hashrate / 1e9).toFixed(0)} EH/s
            </div>
            <div className={`text-sm flex items-center justify-center gap-1 ${
              data.metrics.hashrate_30d_change > 0 ? "text-green-500" : "text-red-500"
            }`}>
              {data.metrics.hashrate_30d_change > 0 ? (
                <TrendingUp className="h-3 w-3" />
              ) : (
                <TrendingDown className="h-3 w-3" />
              )}
              {data.metrics.hashrate_30d_change > 0 ? "+" : ""}
              {data.metrics.hashrate_30d_change.toFixed(1)}% (30d)
            </div>
          </div>
        </div>

        {/* Transfer Value */}
        <div className="mb-4 p-3 bg-muted/30 rounded flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">24h Transfer Value</span>
          </div>
          <span className="font-medium">${formatNumber(data.metrics.transfer_value_usd)}</span>
        </div>

        {/* Signals Row */}
        <div className="flex gap-2 flex-wrap mb-4">
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">MVRV:</span>
            <SignalBadge signal={data.signals.mvrv} />
          </div>
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">NVT:</span>
            <SignalBadge signal={data.signals.nvt} />
          </div>
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">Activity:</span>
            <SignalBadge signal={data.signals.activity} />
          </div>
        </div>

        {/* Recommendation */}
        <div className={`p-4 rounded-lg ${
          data.recommendation.is_undervalued
            ? "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800"
            : data.recommendation.is_overvalued
            ? "bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800"
            : "bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800"
        }`}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="font-medium">Overall Signal</span>
              <SignalBadge signal={data.recommendation.signal} />
            </div>
            <span className="text-sm text-muted-foreground">
              {data.recommendation.confidence}% confidence
            </span>
          </div>
          <p className="text-sm">{data.recommendation.message}</p>

          {data.recommendation.reasoning.length > 0 && (
            <div className="mt-2 space-y-1">
              {data.recommendation.reasoning.map((r, i) => (
                <div key={i} className="text-xs text-muted-foreground">
                  • {r}
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default CryptoOnChainDashboard;
