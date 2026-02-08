"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  RefreshCw,
  ArrowRight,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";

interface AssetPulse {
  asset: string;
  conviction_score: number;
  conviction_label: string;
  confidence: number;
  bullish: number;
  bearish: number;
  headline: string;
  top_driver: string | null;
  error?: string;
}

interface MarketPulseData {
  success: boolean;
  timestamp: string;
  assets: AssetPulse[];
}

interface Props {
  onAssetSelect?: (asset: string) => void;
}

// Asset display names and icons
const ASSET_META: Record<string, { name: string; icon: string; color: string }> = {
  SP500: { name: "S&P 500", icon: "📊", color: "from-blue-500 to-cyan-500" },
  NASDAQ: { name: "NASDAQ", icon: "💻", color: "from-purple-500 to-pink-500" },
  GOLD: { name: "Gold", icon: "🥇", color: "from-yellow-500 to-amber-500" },
  CRUDE: { name: "Crude Oil", icon: "🛢️", color: "from-orange-500 to-red-500" },
  BITCOIN: { name: "Bitcoin", icon: "₿", color: "from-orange-400 to-yellow-500" },
};

export function MarketPulse({ onAssetSelect }: Props) {
  const [data, setData] = useState<MarketPulseData | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/v2/market-pulse");
      const json = await res.json();
      setData(json);
      setLastUpdate(new Date());
    } catch (error) {
      console.error("Failed to fetch market pulse:", error);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
    // Refresh every 5 minutes
    const interval = setInterval(fetchData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  const getConvictionColor = (score: number) => {
    if (score >= 60) return "bg-green-500";
    if (score >= 30) return "bg-green-400";
    if (score <= -60) return "bg-red-500";
    if (score <= -30) return "bg-red-400";
    return "bg-yellow-500";
  };

  const getConvictionGlow = (score: number) => {
    if (score >= 60) return "shadow-green-500/50";
    if (score >= 30) return "shadow-green-400/30";
    if (score <= -60) return "shadow-red-500/50";
    if (score <= -30) return "shadow-red-400/30";
    return "shadow-yellow-500/30";
  };

  const getLabelBadge = (label: string) => {
    const styles: Record<string, string> = {
      "STRONG BUY": "bg-green-500 text-white",
      "BUY": "bg-green-400 text-white",
      "NEUTRAL": "bg-yellow-500 text-black",
      "SELL": "bg-red-400 text-white",
      "STRONG SELL": "bg-red-500 text-white",
    };
    return styles[label] || "bg-gray-500";
  };

  if (loading && !data) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-64">
          <div className="animate-pulse flex items-center gap-2">
            <Activity className="h-5 w-5 animate-spin" />
            <span>Loading Market Pulse...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-purple-500 to-blue-500">
            <Sparkles className="h-6 w-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold">Market Pulse</h2>
            <p className="text-sm text-muted-foreground">
              Real-time signal confluence across all assets
            </p>
          </div>
        </div>
        <Button variant="outline" size="sm" onClick={fetchData} disabled={loading}>
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Asset Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        {data?.assets.map((asset) => {
          const meta = ASSET_META[asset.asset] || { 
            name: asset.asset, 
            icon: "📈", 
            color: "from-gray-500 to-gray-600" 
          };

          if (asset.error) {
            return (
              <Card key={asset.asset} className="opacity-50">
                <CardContent className="pt-4">
                  <div className="text-center">
                    <span className="text-2xl">{meta.icon}</span>
                    <div className="font-medium mt-2">{meta.name}</div>
                    <div className="text-xs text-muted-foreground mt-1">Data unavailable</div>
                  </div>
                </CardContent>
              </Card>
            );
          }

          return (
            <Card 
              key={asset.asset}
              className={`cursor-pointer transition-all hover:scale-105 hover:shadow-lg ${getConvictionGlow(asset.conviction_score)}`}
              onClick={() => onAssetSelect?.(asset.asset)}
            >
              <CardContent className="pt-4">
                {/* Asset Header */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="text-2xl">{meta.icon}</span>
                    <div>
                      <div className="font-semibold">{meta.name}</div>
                      <Badge className={getLabelBadge(asset.conviction_label)} variant="secondary">
                        {asset.conviction_label}
                      </Badge>
                    </div>
                  </div>
                </div>

                {/* Conviction Score */}
                <div className="mb-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-muted-foreground">Conviction</span>
                    <span className={`text-2xl font-bold ${
                      asset.conviction_score > 0 ? "text-green-500" : 
                      asset.conviction_score < 0 ? "text-red-500" : "text-yellow-500"
                    }`}>
                      {asset.conviction_score > 0 ? "+" : ""}{asset.conviction_score.toFixed(0)}
                    </span>
                  </div>
                  
                  {/* Mini gauge */}
                  <div className="relative h-2 bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-full overflow-hidden">
                    <div 
                      className="absolute top-0 bottom-0 w-1 bg-white shadow-lg"
                      style={{ 
                        left: `${((asset.conviction_score + 100) / 200) * 100}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Signal counts */}
                <div className="flex items-center justify-between text-sm mb-3">
                  <div className="flex items-center gap-1 text-green-500">
                    <TrendingUp className="h-3 w-3" />
                    <span>{asset.bullish}</span>
                  </div>
                  <div className="flex items-center gap-1 text-muted-foreground">
                    <span className="text-xs">Confidence: {asset.confidence.toFixed(0)}%</span>
                  </div>
                  <div className="flex items-center gap-1 text-red-500">
                    <span>{asset.bearish}</span>
                    <TrendingDown className="h-3 w-3" />
                  </div>
                </div>

                {/* Top driver */}
                {asset.top_driver && (
                  <div className="text-xs text-muted-foreground border-t pt-2 line-clamp-2">
                    {asset.top_driver}
                  </div>
                )}

                {/* Click indicator */}
                <div className="flex items-center justify-center mt-2 text-xs text-muted-foreground">
                  <span>View Details</span>
                  <ArrowRight className="h-3 w-3 ml-1" />
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Summary Stats */}
      {data && (
        <Card className="bg-gradient-to-r from-slate-800/50 to-slate-900/50">
          <CardContent className="pt-4">
            <div className="grid grid-cols-4 gap-4">
              {/* Most Bullish */}
              <div>
                <div className="text-xs text-muted-foreground mb-1">Most Bullish</div>
                {(() => {
                  const best = [...data.assets]
                    .filter(a => !a.error)
                    .sort((a, b) => b.conviction_score - a.conviction_score)[0];
                  if (!best) return <span>—</span>;
                  return (
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{ASSET_META[best.asset]?.icon}</span>
                      <span className="font-medium">{ASSET_META[best.asset]?.name}</span>
                      <span className="text-green-500 font-bold">+{best.conviction_score.toFixed(0)}</span>
                    </div>
                  );
                })()}
              </div>

              {/* Most Bearish */}
              <div>
                <div className="text-xs text-muted-foreground mb-1">Most Bearish</div>
                {(() => {
                  const worst = [...data.assets]
                    .filter(a => !a.error)
                    .sort((a, b) => a.conviction_score - b.conviction_score)[0];
                  if (!worst) return <span>—</span>;
                  return (
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{ASSET_META[worst.asset]?.icon}</span>
                      <span className="font-medium">{ASSET_META[worst.asset]?.name}</span>
                      <span className="text-red-500 font-bold">{worst.conviction_score.toFixed(0)}</span>
                    </div>
                  );
                })()}
              </div>

              {/* Highest Confidence */}
              <div>
                <div className="text-xs text-muted-foreground mb-1">Highest Confidence</div>
                {(() => {
                  const best = [...data.assets]
                    .filter(a => !a.error)
                    .sort((a, b) => b.confidence - a.confidence)[0];
                  if (!best) return <span>—</span>;
                  return (
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{ASSET_META[best.asset]?.icon}</span>
                      <span className="font-medium">{ASSET_META[best.asset]?.name}</span>
                      <span className="text-blue-500 font-bold">{best.confidence.toFixed(0)}%</span>
                    </div>
                  );
                })()}
              </div>

              {/* Last Update */}
              <div>
                <div className="text-xs text-muted-foreground mb-1">Last Update</div>
                <div className="font-medium">
                  {lastUpdate?.toLocaleTimeString() || "—"}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 text-xs text-muted-foreground">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span>Strong Buy (&gt;60)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-green-400" />
          <span>Buy (30-60)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-yellow-500" />
          <span>Neutral (-30 to 30)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-red-400" />
          <span>Sell (-60 to -30)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span>Strong Sell (&lt;-60)</span>
        </div>
      </div>
    </div>
  );
}

export default MarketPulse;
