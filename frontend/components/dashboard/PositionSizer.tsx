"use client";

import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import {
  Calculator,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Target,
} from "lucide-react";

interface PositionResult {
  asset: string;
  direction: string;
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
  factors: {
    base_size: number;
    regime_mult: number;
    confidence_mult: number;
    cot_adjustment: number;
  };
  reasoning: string[];
  warnings: string[];
}

const ASSETS = [
  { id: "crude-oil", name: "Crude Oil" },
  { id: "gold", name: "Gold" },
  { id: "bitcoin", name: "Bitcoin" },
  { id: "sp500", name: "S&P 500" },
  { id: "natural-gas", name: "Natural Gas" },
];

const PERSONAS = [
  { id: "casual", name: "Casual Investor" },
  { id: "retail", name: "Retail Trader" },
  { id: "wealth_manager", name: "Wealth Manager" },
  { id: "institutional", name: "Institutional" },
  { id: "hedge_fund", name: "Hedge Fund" },
  { id: "hedging", name: "Hedging Desk" },
];

const UrgencyBadge = ({ urgency }: { urgency: string }) => {
  const colors: Record<string, string> = {
    immediate: "bg-red-500",
    today: "bg-orange-500",
    this_week: "bg-yellow-500",
    monitor: "bg-gray-400",
  };

  return (
    <Badge className={`${colors[urgency] || colors.monitor} text-white`}>
      {urgency.replace("_", " ")}
    </Badge>
  );
};

export function PositionSizer() {
  const [asset, setAsset] = useState("crude-oil");
  const [direction, setDirection] = useState("long");
  const [confidence, setConfidence] = useState(70);
  const [winRate, setWinRate] = useState(65);
  const [expectedMove, setExpectedMove] = useState(2.5);
  const [portfolioValue, setPortfolioValue] = useState(100000);
  const [persona, setPersona] = useState("retail");
  
  const [result, setResult] = useState<PositionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const calculatePosition = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch("/api/v2/position", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset,
          direction,
          confidence,
          winRate,
          expectedMove,
          portfolioValue,
          persona,
        }),
      });
      
      const data = await res.json();
      
      if (data.success) {
        setResult(data);
      } else {
        setError(data.error || "Calculation failed");
      }
    } catch (_e) {
      setError("Failed to calculate position size");
    }
    
    setLoading(false);
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Calculator className="h-5 w-5" />
            Position Size Calculator
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {/* Asset */}
            <div className="space-y-2">
              <Label>Asset</Label>
              <Select value={asset} onValueChange={setAsset}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ASSETS.map((a) => (
                    <SelectItem key={a.id} value={a.id}>
                      {a.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Direction */}
            <div className="space-y-2">
              <Label>Direction</Label>
              <Select value={direction} onValueChange={setDirection}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="long">
                    <span className="flex items-center gap-1">
                      <TrendingUp className="h-4 w-4 text-green-500" />
                      Long
                    </span>
                  </SelectItem>
                  <SelectItem value="short">
                    <span className="flex items-center gap-1">
                      <TrendingDown className="h-4 w-4 text-red-500" />
                      Short
                    </span>
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Confidence */}
            <div className="space-y-2">
              <Label>Signal Confidence (%)</Label>
              <Input
                type="number"
                value={confidence}
                onChange={(e) => setConfidence(Number(e.target.value))}
                min={0}
                max={100}
              />
            </div>

            {/* Win Rate */}
            <div className="space-y-2">
              <Label>Historical Win Rate (%)</Label>
              <Input
                type="number"
                value={winRate}
                onChange={(e) => setWinRate(Number(e.target.value))}
                min={0}
                max={100}
              />
            </div>

            {/* Expected Move */}
            <div className="space-y-2">
              <Label>Expected Move (%)</Label>
              <Input
                type="number"
                value={expectedMove}
                onChange={(e) => setExpectedMove(Number(e.target.value))}
                step={0.1}
              />
            </div>

            {/* Portfolio Value */}
            <div className="space-y-2">
              <Label>Portfolio Value ($)</Label>
              <Input
                type="number"
                value={portfolioValue}
                onChange={(e) => setPortfolioValue(Number(e.target.value))}
              />
            </div>

            {/* Persona */}
            <div className="space-y-2">
              <Label>Persona</Label>
              <Select value={persona} onValueChange={setPersona}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PERSONAS.map((p) => (
                    <SelectItem key={p.id} value={p.id}>
                      {p.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Calculate Button */}
            <div className="flex items-end">
              <Button onClick={calculatePosition} disabled={loading} className="w-full">
                {loading ? "Calculating..." : "Calculate"}
              </Button>
            </div>
          </div>

          {error && (
            <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 rounded text-red-600 dark:text-red-400">
              {error}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Results */}
      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              Position Recommendation
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-6">
              {/* Position Size */}
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <div className="text-3xl font-bold text-primary">
                  {result.position_size_pct}%
                </div>
                <div className="text-sm text-muted-foreground">Position Size</div>
                <div className="text-lg font-medium mt-1">
                  ${result.position_value.toLocaleString()}
                </div>
              </div>

              {/* Risk/Reward */}
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <div className="text-3xl font-bold">
                  {result.risk_management.risk_reward.toFixed(1)}x
                </div>
                <div className="text-sm text-muted-foreground">Risk/Reward</div>
                <div className="text-xs mt-1">
                  TP: {result.risk_management.profit_target_pct}% / SL: {result.risk_management.stop_loss_pct}%
                </div>
              </div>

              {/* Max Loss */}
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <div className="text-3xl font-bold text-red-500">
                  ${result.risk_management.max_loss.toLocaleString()}
                </div>
                <div className="text-sm text-muted-foreground">Max Loss</div>
              </div>

              {/* Timing */}
              <div className="text-center p-4 bg-muted/50 rounded-lg">
                <UrgencyBadge urgency={result.timing.urgency} />
                <div className="text-sm text-muted-foreground mt-2">
                  {result.timing.entry_strategy} entry
                </div>
              </div>
            </div>

            {/* Factor Breakdown */}
            <div className="mb-6">
              <h4 className="text-sm font-medium mb-3">Size Factors</h4>
              <div className="grid grid-cols-4 gap-2 text-sm">
                <div className="p-2 bg-muted/30 rounded">
                  <div className="text-muted-foreground text-xs">Base</div>
                  <div className="font-medium">{result.factors.base_size}%</div>
                </div>
                <div className="p-2 bg-muted/30 rounded">
                  <div className="text-muted-foreground text-xs">Regime</div>
                  <div className="font-medium">{result.factors.regime_mult}x</div>
                </div>
                <div className="p-2 bg-muted/30 rounded">
                  <div className="text-muted-foreground text-xs">Confidence</div>
                  <div className="font-medium">{result.factors.confidence_mult}x</div>
                </div>
                <div className="p-2 bg-muted/30 rounded">
                  <div className="text-muted-foreground text-xs">COT Adj</div>
                  <div className="font-medium">
                    {result.factors.cot_adjustment > 0 ? "+" : ""}
                    {result.factors.cot_adjustment}%
                  </div>
                </div>
              </div>
            </div>

            {/* Reasoning */}
            {result.reasoning.length > 0 && (
              <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
                <h4 className="text-sm font-medium text-blue-700 dark:text-blue-400 flex items-center gap-1">
                  <CheckCircle className="h-4 w-4" />
                  Analysis
                </h4>
                <ul className="mt-2 text-sm text-blue-600 dark:text-blue-300 space-y-1">
                  {result.reasoning.map((r, i) => (
                    <li key={i}>• {r}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Warnings */}
            {result.warnings.length > 0 && (
              <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded">
                <h4 className="text-sm font-medium text-yellow-700 dark:text-yellow-400 flex items-center gap-1">
                  <AlertTriangle className="h-4 w-4" />
                  Warnings
                </h4>
                <ul className="mt-2 text-sm text-yellow-600 dark:text-yellow-300 space-y-1">
                  {result.warnings.map((w, i) => (
                    <li key={i}>• {w}</li>
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

export default PositionSizer;
