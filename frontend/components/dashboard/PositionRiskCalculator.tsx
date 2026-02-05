"use client";

import { useState, useMemo, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  Shield,
  AlertTriangle,
  DollarSign,
  TrendingUp,
  TrendingDown,
  Target,
  Percent,
  Scale,
  ChevronRight,
  Info,
  Gauge,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface PositionConfig {
  accountSize: number;
  riskPercent: number;
  entryPrice: number;
  stopLoss: number;
  takeProfitLevels: number[];
  leverage?: number;
}

interface CalculatedMetrics {
  positionSize: number;
  positionValue: number;
  riskAmount: number;
  rewardAmounts: number[];
  riskRewardRatios: number[];
  stopLossPercent: number;
  leveragedExposure: number;
  maxLoss: number;
  potentialGains: number[];
}

interface PositionRiskCalculatorProps {
  initialConfig?: Partial<PositionConfig>;
  assetSymbol?: string;
  assetName?: string;
  currentPrice?: number;
  direction?: "long" | "short";
  onConfigChange?: (config: PositionConfig, metrics: CalculatedMetrics) => void;
}

// ============================================================================
// Calculation Functions
// ============================================================================

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function calculateMetrics(config: PositionConfig, direction: "long" | "short"): CalculatedMetrics {
  const {
    accountSize,
    riskPercent,
    entryPrice,
    stopLoss,
    takeProfitLevels,
    leverage = 1,
  } = config;

  // Risk amount in dollars
  const riskAmount = accountSize * (riskPercent / 100);

  // Stop loss distance
  const stopLossDistance = Math.abs(entryPrice - stopLoss);
  const stopLossPercent = (stopLossDistance / entryPrice) * 100;

  // Position size (number of units/contracts)
  const positionSize = riskAmount / stopLossDistance;
  const positionValue = positionSize * entryPrice;

  // Leveraged exposure
  const leveragedExposure = positionValue * leverage;

  // Maximum loss (should equal risk amount)
  const maxLoss = positionSize * stopLossDistance;

  // Take profit calculations
  const rewardAmounts = takeProfitLevels.map((tp) => {
    const tpDistance = Math.abs(tp - entryPrice);
    return positionSize * tpDistance;
  });

  const riskRewardRatios = takeProfitLevels.map((tp) => {
    const tpDistance = Math.abs(tp - entryPrice);
    return tpDistance / stopLossDistance;
  });

  const potentialGains = takeProfitLevels.map((tp) => {
    const tpDistance = Math.abs(tp - entryPrice);
    return positionSize * tpDistance;
  });

  return {
    positionSize,
    positionValue,
    riskAmount,
    rewardAmounts,
    riskRewardRatios,
    stopLossPercent,
    leveragedExposure,
    maxLoss,
    potentialGains,
  };
}

// ============================================================================
// Slider Component
// ============================================================================

function RiskSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  unit,
  color = "blue",
  icon: Icon,
  formatValue,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  unit?: string;
  color?: "blue" | "green" | "red" | "amber" | "purple";
  icon?: React.ElementType;
  formatValue?: (value: number) => string;
}) {
  const percentage = ((value - min) / (max - min)) * 100;

  const colorClasses = {
    blue: { bg: "bg-blue-500", text: "text-blue-400", track: "from-blue-500/50 to-blue-500" },
    green: { bg: "bg-green-500", text: "text-green-400", track: "from-green-500/50 to-green-500" },
    red: { bg: "bg-red-500", text: "text-red-400", track: "from-red-500/50 to-red-500" },
    amber: { bg: "bg-amber-500", text: "text-amber-400", track: "from-amber-500/50 to-amber-500" },
    purple: { bg: "bg-purple-500", text: "text-purple-400", track: "from-purple-500/50 to-purple-500" },
  };

  const colors = colorClasses[color];
  const displayValue = formatValue ? formatValue(value) : value.toFixed(2);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {Icon && <Icon className={cn("w-4 h-4", colors.text)} />}
          <span className="text-xs font-medium text-neutral-400">{label}</span>
        </div>
        <span className={cn("text-sm font-mono font-bold", colors.text)}>
          {displayValue}
          {unit && <span className="text-neutral-500 ml-1">{unit}</span>}
        </span>
      </div>
      <div className="relative h-2 bg-neutral-800 rounded-full overflow-hidden">
        <div
          className={cn("absolute h-full rounded-full bg-gradient-to-r", colors.track)}
          style={{ width: `${percentage}%` }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
        />
        {/* Thumb indicator */}
        <div
          className={cn(
            "absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full border-2 border-neutral-900",
            colors.bg,
            "shadow-lg transition-transform hover:scale-110"
          )}
          style={{ left: `calc(${percentage}% - 8px)` }}
        />
      </div>
    </div>
  );
}

// ============================================================================
// Risk Level Indicator
// ============================================================================

function RiskLevelIndicator({
  riskPercent,
  riskRewardRatio,
}: {
  riskPercent: number;
  riskRewardRatio: number;
}) {
  // Determine overall risk level
  let riskLevel: "conservative" | "moderate" | "aggressive" | "extreme";
  let riskColor: string;
  let riskLabel: string;

  if (riskPercent <= 1 && riskRewardRatio >= 2) {
    riskLevel = "conservative";
    riskColor = "text-green-400";
    riskLabel = "Conservative";
  } else if (riskPercent <= 2 && riskRewardRatio >= 1.5) {
    riskLevel = "moderate";
    riskColor = "text-blue-400";
    riskLabel = "Moderate";
  } else if (riskPercent <= 3 && riskRewardRatio >= 1) {
    riskLevel = "aggressive";
    riskColor = "text-amber-400";
    riskLabel = "Aggressive";
  } else {
    riskLevel = "extreme";
    riskColor = "text-red-400";
    riskLabel = "High Risk";
  }

  const segments = [
    { label: "Conservative", color: "#22c55e", threshold: 0.25 },
    { label: "Moderate", color: "#3b82f6", threshold: 0.5 },
    { label: "Aggressive", color: "#f59e0b", threshold: 0.75 },
    { label: "Extreme", color: "#ef4444", threshold: 1 },
  ];

  // Normalize risk score (0 to 1)
  const riskScore = Math.min(1, (riskPercent / 5) * 0.5 + (1 - Math.min(riskRewardRatio, 3) / 3) * 0.5);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Gauge className={cn("w-4 h-4", riskColor)} />
          <span className="text-xs font-medium text-neutral-400">Risk Assessment</span>
        </div>
        <Badge
          className={cn(
            "text-xs font-bold",
            riskLevel === "conservative"
              ? "bg-green-500/10 border-green-500/30 text-green-400"
              : riskLevel === "moderate"
              ? "bg-blue-500/10 border-blue-500/30 text-blue-400"
              : riskLevel === "aggressive"
              ? "bg-amber-500/10 border-amber-500/30 text-amber-400"
              : "bg-red-500/10 border-red-500/30 text-red-400"
          )}
        >
          {riskLevel === "extreme" && <AlertTriangle className="w-3 h-3 mr-1" />}
          {riskLabel}
        </Badge>
      </div>

      {/* Risk gauge */}
      <div className="relative h-3 bg-neutral-800 rounded-full overflow-hidden">
        {/* Gradient track */}
        <div className="absolute inset-0 flex">
          {segments.map((seg, i) => (
            <div
              key={i}
              className="h-full flex-1"
              style={{ backgroundColor: `${seg.color}30` }}
            />
          ))}
        </div>
        {/* Indicator */}
        <div
          className="absolute top-0 bottom-0 w-1 bg-white shadow-lg transition-all duration-300"
          style={{ left: `${riskScore * 100}%` }}
        />
      </div>

      {/* Labels */}
      <div className="flex justify-between text-[10px] text-neutral-600">
        {segments.map((seg) => (
          <span key={seg.label}>{seg.label}</span>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Scenario Card
// ============================================================================

function ScenarioCard({
  label,
  price,
  pnl,
  pnlPercent,
  isProfit,
  probability,
  ratio,
}: {
  label: string;
  price: number;
  pnl: number;
  pnlPercent: number;
  isProfit: boolean;
  probability?: number;
  ratio?: number;
}) {
  return (
    <div
      className={cn(
        "p-3 rounded-lg border transition-all hover:scale-[1.02]",
        isProfit
          ? "bg-green-500/5 border-green-500/20 hover:border-green-500/40"
          : "bg-red-500/5 border-red-500/20 hover:border-red-500/40"
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-neutral-500">{label}</span>
        {ratio !== undefined && (
          <span
            className={cn(
              "text-[10px] font-mono px-1.5 py-0.5 rounded",
              isProfit ? "bg-green-500/10 text-green-400" : "bg-red-500/10 text-red-400"
            )}
          >
            {isProfit ? `1:${ratio.toFixed(1)}` : "Risk"}
          </span>
        )}
      </div>

      <div className="space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-xs text-neutral-400">Price</span>
          <span className="text-sm font-mono text-neutral-200">
            ${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-neutral-400">P&L</span>
          <span
            className={cn(
              "text-sm font-mono font-bold",
              isProfit ? "text-green-400" : "text-red-400"
            )}
          >
            {isProfit ? "+" : "-"}${Math.abs(pnl).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-neutral-400">Return</span>
          <span
            className={cn(
              "text-xs font-mono",
              isProfit ? "text-green-400" : "text-red-400"
            )}
          >
            {isProfit ? "+" : ""}{pnlPercent.toFixed(2)}%
          </span>
        </div>
        {probability !== undefined && (
          <div className="flex items-center justify-between pt-1 border-t border-neutral-800">
            <span className="text-xs text-neutral-500">Probability</span>
            <span className="text-xs font-mono text-neutral-400">{probability}%</span>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function PositionRiskCalculator({
  initialConfig,
  assetSymbol = "CL",
  assetName = "Crude Oil",
  currentPrice = 73.5,
  direction = "long",
  onConfigChange,
}: PositionRiskCalculatorProps) {
  const [config, setConfig] = useState<PositionConfig>({
    accountSize: initialConfig?.accountSize ?? 100000,
    riskPercent: initialConfig?.riskPercent ?? 1,
    entryPrice: initialConfig?.entryPrice ?? currentPrice,
    stopLoss: initialConfig?.stopLoss ?? (direction === "long" ? currentPrice * 0.975 : currentPrice * 1.025),
    takeProfitLevels: initialConfig?.takeProfitLevels ?? [
      direction === "long" ? currentPrice * 1.015 : currentPrice * 0.985,
      direction === "long" ? currentPrice * 1.03 : currentPrice * 0.97,
      direction === "long" ? currentPrice * 1.05 : currentPrice * 0.95,
    ],
    leverage: initialConfig?.leverage ?? 1,
  });

  const metrics = useMemo(() => calculateMetrics(config, direction), [config, direction]);

  const updateConfig = useCallback(
    (updates: Partial<PositionConfig>) => {
      const newConfig = { ...config, ...updates };
      setConfig(newConfig);
      onConfigChange?.(newConfig, calculateMetrics(newConfig, direction));
    },
    [config, direction, onConfigChange]
  );

  const isLong = direction === "long";

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-500/10 rounded-lg border border-purple-500/20">
              <Scale className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <CardTitle className="text-sm font-semibold text-neutral-200">
                Position Calculator
              </CardTitle>
              <div className="flex items-center gap-2 mt-0.5">
                <Badge
                  className={cn(
                    "text-xs",
                    isLong
                      ? "bg-green-500/10 border-green-500/30 text-green-400"
                      : "bg-red-500/10 border-red-500/30 text-red-400"
                  )}
                >
                  {isLong ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
                  {direction.toUpperCase()}
                </Badge>
                <span className="text-xs text-neutral-500">
                  {assetSymbol} - {assetName}
                </span>
              </div>
            </div>
          </div>

          <div className="text-right">
            <div className="text-xs text-neutral-500">Current Price</div>
            <div className="text-lg font-mono font-bold text-neutral-200">
              ${currentPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-4 pt-0 space-y-6">
        {/* Input Sliders */}
        <div className="space-y-4">
          <RiskSlider
            label="Account Size"
            value={config.accountSize}
            onChange={(v) => updateConfig({ accountSize: v })}
            min={10000}
            max={1000000}
            step={10000}
            color="blue"
            icon={DollarSign}
            formatValue={(v) => `$${(v / 1000).toFixed(0)}K`}
          />

          <RiskSlider
            label="Risk Per Trade"
            value={config.riskPercent}
            onChange={(v) => updateConfig({ riskPercent: v })}
            min={0.25}
            max={5}
            step={0.25}
            unit="%"
            color={config.riskPercent <= 1 ? "green" : config.riskPercent <= 2 ? "amber" : "red"}
            icon={Percent}
          />

          <RiskSlider
            label="Entry Price"
            value={config.entryPrice}
            onChange={(v) => updateConfig({ entryPrice: v })}
            min={currentPrice * 0.95}
            max={currentPrice * 1.05}
            step={0.01}
            color="purple"
            icon={Target}
            formatValue={(v) => `$${v.toFixed(2)}`}
          />

          <RiskSlider
            label="Stop Loss"
            value={config.stopLoss}
            onChange={(v) => updateConfig({ stopLoss: v })}
            min={isLong ? currentPrice * 0.9 : currentPrice * 1.01}
            max={isLong ? currentPrice * 0.99 : currentPrice * 1.1}
            step={0.01}
            color="red"
            icon={Shield}
            formatValue={(v) => `$${v.toFixed(2)}`}
          />
        </div>

        {/* Risk Assessment */}
        <RiskLevelIndicator
          riskPercent={config.riskPercent}
          riskRewardRatio={metrics.riskRewardRatios[0] || 1}
        />

        {/* Calculated Metrics */}
        <div className="grid grid-cols-2 gap-3">
          <div className="p-3 bg-neutral-800/50 rounded-lg border border-neutral-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <Info className="w-3 h-3 text-neutral-500" />
              <span className="text-[10px] uppercase tracking-wider text-neutral-500">
                Position Size
              </span>
            </div>
            <div className="font-mono text-lg font-bold text-blue-400">
              {metrics.positionSize.toFixed(2)}
              <span className="text-xs text-neutral-500 ml-1">units</span>
            </div>
          </div>

          <div className="p-3 bg-neutral-800/50 rounded-lg border border-neutral-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <DollarSign className="w-3 h-3 text-neutral-500" />
              <span className="text-[10px] uppercase tracking-wider text-neutral-500">
                Position Value
              </span>
            </div>
            <div className="font-mono text-lg font-bold text-neutral-200">
              ${metrics.positionValue.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
          </div>

          <div className="p-3 bg-neutral-800/50 rounded-lg border border-neutral-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <AlertTriangle className="w-3 h-3 text-red-500" />
              <span className="text-[10px] uppercase tracking-wider text-neutral-500">
                Max Loss
              </span>
            </div>
            <div className="font-mono text-lg font-bold text-red-400">
              -${metrics.maxLoss.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
          </div>

          <div className="p-3 bg-neutral-800/50 rounded-lg border border-neutral-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <Target className="w-3 h-3 text-green-500" />
              <span className="text-[10px] uppercase tracking-wider text-neutral-500">
                Max Gain (T3)
              </span>
            </div>
            <div className="font-mono text-lg font-bold text-green-400">
              +${(metrics.potentialGains[2] || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </div>
          </div>
        </div>

        {/* Scenario Cards */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-medium text-neutral-400 flex items-center gap-2">
              <ChevronRight className="w-3.5 h-3.5" />
              Profit/Loss Scenarios
            </span>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
            {/* Stop Loss */}
            <ScenarioCard
              label="Stop Loss"
              price={config.stopLoss}
              pnl={metrics.maxLoss}
              pnlPercent={(metrics.maxLoss / config.accountSize) * 100}
              isProfit={false}
            />

            {/* Take Profits */}
            {config.takeProfitLevels.map((tp, i) => (
              <ScenarioCard
                key={i}
                label={`Target ${i + 1}`}
                price={tp}
                pnl={metrics.potentialGains[i]}
                pnlPercent={(metrics.potentialGains[i] / config.accountSize) * 100}
                isProfit={true}
                ratio={metrics.riskRewardRatios[i]}
              />
            ))}
          </div>
        </div>

        {/* Summary Footer */}
        <div className="flex items-center justify-between p-3 bg-gradient-to-r from-purple-500/10 to-blue-500/10 rounded-lg border border-purple-500/20">
          <div className="flex items-center gap-2">
            <Scale className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-neutral-300">
              Risking{" "}
              <span className="font-mono font-bold text-red-400">
                ${metrics.riskAmount.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </span>{" "}
              ({config.riskPercent}%) for potential{" "}
              <span className="font-mono font-bold text-green-400">
                ${(metrics.potentialGains[0] || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </span>{" "}
              gain
            </span>
          </div>
          <Badge className="bg-blue-500/10 border-blue-500/30 text-blue-400 font-mono">
            R:R 1:{metrics.riskRewardRatios[0]?.toFixed(1) || "N/A"}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}
