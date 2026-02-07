"use client";

import { useEffect, useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Activity,
  AlertTriangle,
  Shield,
  Zap,
  TrendingUp,
  Clock,
  Gauge,
  RefreshCw,
} from "lucide-react";
import { useQuantumDashboard } from "@/hooks/useApi";
import type { QuantumRegime, QuantumAssetRegime } from "@/lib/api-client";

// Re-export for backwards compatibility - aligned with API types
export type QuantumRegimeState = QuantumRegime;

export interface QuantumRegimeData {
  regime: QuantumRegimeState;
  entropy: number; // 0-1 scale (derived from realized_vol)
  confidence: number; // 0-1 scale
  timeInRegime: number; // seconds
  lastTransition: string; // ISO timestamp
  asset?: string;
}

// Regime configuration with institutional color coding
// Aligned with API types: LOW_VOL | NORMAL | ELEVATED | CRISIS
const REGIME_CONFIG: Record<
  QuantumRegimeState,
  {
    label: string;
    description: string;
    color: string;
    bgColor: string;
    borderColor: string;
    glowColor: string;
    icon: typeof Activity;
  }
> = {
  LOW_VOL: {
    label: "Low Volatility",
    description: "Market conditions stable, reduced risk environment",
    color: "#22c55e",
    bgColor: "bg-green-500/10",
    borderColor: "border-green-500/30",
    glowColor: "rgba(34, 197, 94, 0.4)",
    icon: Shield,
  },
  NORMAL: {
    label: "Normal",
    description: "Standard market conditions, balanced opportunity-risk",
    color: "#3b82f6",
    bgColor: "bg-blue-500/10",
    borderColor: "border-blue-500/30",
    glowColor: "rgba(59, 130, 246, 0.4)",
    icon: TrendingUp,
  },
  ELEVATED: {
    label: "Elevated",
    description: "Increased market turbulence, enhanced risk management advised",
    color: "#f59e0b",
    bgColor: "bg-amber-500/10",
    borderColor: "border-amber-500/30",
    glowColor: "rgba(245, 158, 11, 0.4)",
    icon: AlertTriangle,
  },
  CRISIS: {
    label: "Crisis",
    description: "Extreme market stress, defensive positioning recommended",
    color: "#ef4444",
    bgColor: "bg-red-500/10",
    borderColor: "border-red-500/30",
    glowColor: "rgba(239, 68, 68, 0.4)",
    icon: Zap,
  },
};

interface QuantumStatusWidgetProps {
  data?: QuantumRegimeData;
  asset?: string;
  className?: string;
  size?: "sm" | "md" | "lg";
  showDetails?: boolean;
  onRefresh?: () => void;
  isLoading?: boolean;
}

// Circular entropy gauge component
function EntropyGauge({
  value,
  size,
  color,
  animated = true,
}: {
  value: number;
  size: "sm" | "md" | "lg";
  color: string;
  animated?: boolean;
}) {
  const [animatedValue, setAnimatedValue] = useState(animated ? 0 : value);

  useEffect(() => {
    if (!animated) {
      setAnimatedValue(value);
      return;
    }
    const timer = setTimeout(() => setAnimatedValue(value), 100);
    return () => clearTimeout(timer);
  }, [value, animated]);

  const sizes = {
    sm: { width: 64, height: 64, strokeWidth: 6, radius: 24, fontSize: 10 },
    md: { width: 88, height: 88, strokeWidth: 7, radius: 34, fontSize: 12 },
    lg: { width: 120, height: 120, strokeWidth: 8, radius: 48, fontSize: 14 },
  };

  const { width, height, strokeWidth, radius, fontSize } = sizes[size];
  const centerX = width / 2;
  const centerY = height / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference * (1 - animatedValue);

  return (
    <div className="relative flex items-center justify-center">
      <svg width={width} height={height} className="transform -rotate-90">
        <defs>
          {/* Gradient for filled portion */}
          <linearGradient id={`entropy-gradient-${size}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={color} stopOpacity={0.6} />
            <stop offset="100%" stopColor={color} stopOpacity={1} />
          </linearGradient>
          {/* Glow filter */}
          <filter id={`entropy-glow-${size}`} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background circle */}
        <circle
          cx={centerX}
          cy={centerY}
          r={radius}
          fill="none"
          stroke="#262626"
          strokeWidth={strokeWidth}
        />

        {/* Filled portion */}
        <circle
          cx={centerX}
          cy={centerY}
          r={radius}
          fill="none"
          stroke={`url(#entropy-gradient-${size})`}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          filter={`url(#entropy-glow-${size})`}
          className="transition-all duration-1000 ease-out"
        />

        {/* Tick marks */}
        {[0, 0.25, 0.5, 0.75, 1].map((tick, i) => {
          const angle = tick * 360 - 90;
          const rad = (angle * Math.PI) / 180;
          const innerR = radius - strokeWidth / 2 - 3;
          const outerR = radius - strokeWidth / 2 + 1;
          const x1 = centerX + innerR * Math.cos(rad);
          const y1 = centerY + innerR * Math.sin(rad);
          const x2 = centerX + outerR * Math.cos(rad);
          const y2 = centerY + outerR * Math.sin(rad);
          return (
            <line
              key={i}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="#525252"
              strokeWidth={1}
              className="transform rotate-90"
              style={{ transformOrigin: `${centerX}px ${centerY}px` }}
            />
          );
        })}
      </svg>

      {/* Center value display */}
      <div
        className="absolute inset-0 flex flex-col items-center justify-center"
        style={{ color }}
      >
        <span
          className="font-mono font-bold"
          style={{ fontSize: fontSize + 4 }}
        >
          {(animatedValue * 100).toFixed(0)}
        </span>
        <span className="text-neutral-500 uppercase tracking-wider" style={{ fontSize: fontSize - 3 }}>
          Entropy
        </span>
      </div>
    </div>
  );
}

// Confidence bar component
function ConfidenceIndicator({
  value,
  size,
  color,
}: {
  value: number;
  size: "sm" | "md" | "lg";
  color: string;
}) {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedValue(value), 150);
    return () => clearTimeout(timer);
  }, [value]);

  const heights = { sm: "h-1", md: "h-1.5", lg: "h-2" };
  const widths = { sm: "w-16", md: "w-24", lg: "w-32" };

  const getConfidenceLabel = (v: number): string => {
    if (v >= 0.9) return "Very High";
    if (v >= 0.7) return "High";
    if (v >= 0.5) return "Moderate";
    if (v >= 0.3) return "Low";
    return "Very Low";
  };

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-neutral-500">
          Confidence
        </span>
        <span className="text-xs font-mono" style={{ color }}>
          {(animatedValue * 100).toFixed(0)}%
        </span>
      </div>
      <div className={cn("bg-neutral-800 rounded-full overflow-hidden", heights[size], widths[size])}>
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${animatedValue * 100}%`,
            backgroundColor: color,
            boxShadow: `0 0 8px ${color}40`,
          }}
        />
      </div>
      <span className="text-[10px] text-neutral-500">
        {getConfidenceLabel(animatedValue)}
      </span>
    </div>
  );
}

// Time in regime formatter
function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  if (hours < 24) return `${hours}h ${mins}m`;
  const days = Math.floor(hours / 24);
  return `${days}d ${hours % 24}h`;
}

// Asset symbol to API name mapping
const ASSET_NAME_MAP: Record<string, string> = {
  CL: "Crude_Oil",
  BTC: "Bitcoin",
  GC: "GOLD",
  NG: "Natural_Gas",
  // Also support direct names
  "Crude_Oil": "Crude_Oil",
  "Bitcoin": "Bitcoin",
  "GOLD": "GOLD",
  "Natural_Gas": "Natural_Gas",
};

// Transform API regime data to component format
function transformRegimeData(
  assetRegime: QuantumAssetRegime | undefined,
  asset: string,
  timestamp: string
): QuantumRegimeData {
  if (!assetRegime) {
    // Fallback when no data available
    return {
      regime: "NORMAL",
      entropy: 0.5,
      confidence: 0.5,
      timeInRegime: 0,
      lastTransition: new Date().toISOString(),
      asset,
    };
  }

  // Convert realized_vol to entropy (0-1 scale, higher vol = higher entropy)
  // Typical vol ranges: 10-50% annualized, normalize to 0-1
  const normalizedVol = Math.min(assetRegime.realized_vol / 50, 1);

  return {
    regime: assetRegime.regime,
    entropy: normalizedVol,
    confidence: assetRegime.confidence,
    timeInRegime: 3600, // API doesn't provide this, assume 1 hour default
    lastTransition: timestamp,
    asset,
  };
}

export function QuantumStatusWidget({
  data: externalData,
  asset = "BTC",
  className,
  size = "md",
  showDetails = true,
  onRefresh,
  isLoading: externalLoading = false,
}: QuantumStatusWidgetProps) {
  // Fetch real quantum data from API
  const { data: quantumData, isLoading: apiLoading, refetch } = useQuantumDashboard();

  // Determine loading state
  const isLoading = externalLoading || apiLoading;

  // Get asset-specific regime from API data
  const data = useMemo(() => {
    if (externalData) return externalData;

    // Map asset symbol to API name
    const apiAssetName = ASSET_NAME_MAP[asset] || asset;

    // Find regime for this asset
    const assetRegime = quantumData?.regimes?.[apiAssetName];

    return transformRegimeData(
      assetRegime,
      asset,
      quantumData?.timestamp || new Date().toISOString()
    );
  }, [externalData, quantumData, asset]);

  // Handle refresh with API refetch
  const handleRefresh = () => {
    if (onRefresh) {
      onRefresh();
    } else {
      refetch();
    }
  };

  const config = REGIME_CONFIG[data.regime];
  const RegimeIcon = config.icon;

  const sizeClasses = {
    sm: "p-3",
    md: "p-4",
    lg: "p-5",
  };

  const iconSizes = {
    sm: "w-4 h-4",
    md: "w-5 h-5",
    lg: "w-6 h-6",
  };

  if (isLoading) {
    return (
      <Card className={cn("bg-neutral-900/50 border-neutral-800", className)}>
        <CardContent className={cn(sizeClasses[size], "flex items-center justify-center gap-3")}>
          <RefreshCw className="w-5 h-5 text-neutral-500 animate-spin" />
          <span className="text-sm text-neutral-500">Loading quantum state...</span>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card
      className={cn(
        "bg-neutral-900/50 border-neutral-800 overflow-hidden transition-all duration-300",
        "hover:border-neutral-700",
        className
      )}
      style={{
        boxShadow: `0 0 20px ${config.glowColor}`,
      }}
    >
      <CardHeader className={cn("pb-2", sizeClasses[size])}>
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            <Gauge className={cn(iconSizes[size], "text-neutral-400")} />
            Quantum Regime
            {asset && (
              <span className="text-xs font-mono text-neutral-500 bg-neutral-800 px-2 py-0.5 rounded">
                {asset}
              </span>
            )}
          </CardTitle>
          <button
            onClick={handleRefresh}
            className="p-1.5 rounded-lg hover:bg-neutral-800 transition-colors"
            title="Refresh quantum state"
          >
            <RefreshCw className={cn(
              "w-3.5 h-3.5 text-neutral-500 hover:text-neutral-300",
              apiLoading && "animate-spin"
            )} />
          </button>
        </div>
      </CardHeader>

      <CardContent className={cn("pt-0", sizeClasses[size])}>
        <div className="flex items-start gap-4">
          {/* Entropy Gauge */}
          <div className="flex-shrink-0">
            <EntropyGauge
              value={data.entropy}
              size={size}
              color={config.color}
            />
          </div>

          {/* Regime Info */}
          <div className="flex-1 min-w-0 space-y-3">
            {/* Regime Badge */}
            <div
              className={cn(
                "inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border",
                config.bgColor,
                config.borderColor
              )}
            >
              <RegimeIcon className={iconSizes[size]} style={{ color: config.color }} />
              <span
                className="text-sm font-semibold tracking-wide"
                style={{ color: config.color }}
              >
                {config.label}
              </span>
            </div>

            {showDetails && (
              <>
                {/* Description */}
                <p className="text-xs text-neutral-500 leading-relaxed">
                  {config.description}
                </p>

                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-3">
                  {/* Confidence */}
                  <ConfidenceIndicator
                    value={data.confidence}
                    size={size}
                    color={config.color}
                  />

                  {/* Time in Regime */}
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3 text-neutral-500" />
                      <span className="text-[10px] uppercase tracking-wider text-neutral-500">
                        Time in Regime
                      </span>
                    </div>
                    <span
                      className="text-sm font-mono font-semibold"
                      style={{ color: config.color }}
                    >
                      {formatDuration(data.timeInRegime)}
                    </span>
                    <span className="text-[10px] text-neutral-600">
                      Since {new Date(data.lastTransition).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </span>
                  </div>
                </div>
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default QuantumStatusWidget;
