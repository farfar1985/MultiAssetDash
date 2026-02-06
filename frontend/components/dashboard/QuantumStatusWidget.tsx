"use client";

import { useQuantumDashboard } from "@/hooks/useApi";
import { cn } from "@/lib/utils";
import type { QuantumRegime, ContagionLevel } from "@/lib/api-client";
import {
  Activity,
  AlertTriangle,
  Shield,
  Zap,
  TrendingDown,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";

// Regime configuration
const REGIME_CONFIG: Record<
  QuantumRegime,
  { label: string; color: string; bg: string; border: string; icon: typeof Activity }
> = {
  LOW_VOL: {
    label: "Low Volatility",
    color: "text-green-400",
    bg: "bg-green-500/10",
    border: "border-green-500/30",
    icon: Shield,
  },
  NORMAL: {
    label: "Normal",
    color: "text-blue-400",
    bg: "bg-blue-500/10",
    border: "border-blue-500/30",
    icon: Activity,
  },
  ELEVATED: {
    label: "Elevated",
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    border: "border-amber-500/30",
    icon: AlertTriangle,
  },
  CRISIS: {
    label: "Crisis",
    color: "text-red-400",
    bg: "bg-red-500/10",
    border: "border-red-500/30",
    icon: Zap,
  },
};

// Contagion level configuration
const CONTAGION_CONFIG: Record<
  ContagionLevel,
  { label: string; color: string; severity: number }
> = {
  LOW: { label: "Low", color: "text-green-400", severity: 1 },
  MODERATE: { label: "Moderate", color: "text-amber-400", severity: 2 },
  HIGH: { label: "High", color: "text-orange-400", severity: 3 },
  CRITICAL: { label: "Critical", color: "text-red-400", severity: 4 },
};

interface QuantumStatusWidgetProps {
  className?: string;
  compact?: boolean;
}

export function QuantumStatusWidget({
  className,
  compact = false,
}: QuantumStatusWidgetProps) {
  const { data, isLoading, isError, refetch } = useQuantumDashboard();

  // Determine dominant regime across all assets
  const getDominantRegime = (): QuantumRegime | null => {
    if (!data?.regimes) return null;
    const regimes = Object.values(data.regimes);
    if (regimes.length === 0) return null;

    // Priority: CRISIS > ELEVATED > NORMAL > LOW_VOL
    const priority: QuantumRegime[] = ["CRISIS", "ELEVATED", "NORMAL", "LOW_VOL"];
    for (const regime of priority) {
      if (regimes.some((r) => r.regime === regime)) {
        return regime;
      }
    }
    return "NORMAL";
  };

  // Get warnings based on regime and contagion
  const getWarnings = (): string[] => {
    const warnings: string[] = [];
    if (!data) return warnings;

    // Check for crisis or elevated regimes
    if (data.regimes) {
      const crisisAssets = Object.entries(data.regimes)
        .filter(([, r]) => r.regime === "CRISIS")
        .map(([name]) => name);
      const elevatedAssets = Object.entries(data.regimes)
        .filter(([, r]) => r.regime === "ELEVATED")
        .map(([name]) => name);

      if (crisisAssets.length > 0) {
        warnings.push(`Crisis regime: ${crisisAssets.join(", ")}`);
      }
      if (elevatedAssets.length > 0) {
        warnings.push(`Elevated volatility: ${elevatedAssets.join(", ")}`);
      }
    }

    // Check contagion
    if (data.contagion) {
      if (data.contagion.level === "CRITICAL") {
        warnings.push("Critical cross-asset contagion detected");
      } else if (data.contagion.level === "HIGH") {
        warnings.push("High contagion risk between assets");
      }
      if (data.contagion.highest_risk_pair) {
        const [a1, a2] = data.contagion.highest_risk_pair;
        warnings.push(`Highest risk: ${a1} <-> ${a2}`);
      }
    }

    return warnings;
  };

  if (isLoading) {
    return (
      <div
        className={cn(
          "flex items-center gap-2 px-3 py-2 rounded-lg border bg-neutral-800/50 border-neutral-700/50",
          className
        )}
      >
        <Loader2 className="w-4 h-4 text-neutral-400 animate-spin" />
        <span className="text-xs text-neutral-400">Loading quantum status...</span>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div
        className={cn(
          "flex items-center gap-2 px-3 py-2 rounded-lg border bg-neutral-800/50 border-neutral-700/50 cursor-pointer hover:bg-neutral-800",
          className
        )}
        onClick={() => refetch()}
        title="Click to retry connection"
      >
        <AlertTriangle className="w-4 h-4 text-amber-400" />
        <span className="text-xs text-neutral-400">Quantum API offline</span>
        <RefreshCw className="w-3 h-3 text-neutral-500" />
      </div>
    );
  }

  const dominantRegime = getDominantRegime();
  const regimeConfig = dominantRegime ? REGIME_CONFIG[dominantRegime] : null;
  const contagion = data.contagion;
  const contagionConfig = contagion ? CONTAGION_CONFIG[contagion.level] : null;
  const warnings = getWarnings();
  const RegimeIcon = regimeConfig?.icon || Activity;

  // Build tooltip text
  const tooltipText = [
    `Regime: ${regimeConfig?.label || "Unknown"}`,
    contagionConfig ? `Contagion: ${contagionConfig.label}` : null,
    warnings.length > 0 ? `Warnings: ${warnings.join("; ")}` : null,
  ]
    .filter(Boolean)
    .join("\n");

  if (compact) {
    return (
      <div
        className={cn(
          "flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-colors cursor-pointer hover:opacity-80",
          regimeConfig?.bg || "bg-neutral-800/50",
          regimeConfig?.border || "border-neutral-700/50",
          className
        )}
        onClick={() => refetch()}
        title={tooltipText}
      >
        <RegimeIcon
          className={cn("w-3.5 h-3.5", regimeConfig?.color || "text-neutral-400")}
        />
        <span className={cn("text-xs font-medium", regimeConfig?.color)}>
          {regimeConfig?.label || "Unknown"}
        </span>
        {warnings.length > 0 && (
          <div className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
        )}
      </div>
    );
  }

  // Full widget view
  return (
    <div
      className={cn(
        "flex items-center gap-4 px-4 py-2.5 rounded-xl border transition-all",
        regimeConfig?.bg || "bg-neutral-800/50",
        regimeConfig?.border || "border-neutral-700/50",
        className
      )}
    >
      {/* Regime Status */}
      <div className="flex items-center gap-2">
        <div
          className={cn(
            "p-1.5 rounded-lg",
            regimeConfig?.bg || "bg-neutral-800"
          )}
        >
          <RegimeIcon
            className={cn("w-4 h-4", regimeConfig?.color || "text-neutral-400")}
          />
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">
            Regime
          </div>
          <div className={cn("text-sm font-semibold", regimeConfig?.color)}>
            {regimeConfig?.label || "Unknown"}
          </div>
        </div>
      </div>

      <div className="w-px h-8 bg-neutral-700/50" />

      {/* Contagion Level */}
      <div className="flex items-center gap-2">
        <TrendingDown
          className={cn(
            "w-4 h-4",
            contagionConfig?.color || "text-neutral-400"
          )}
        />
        <div>
          <div className="text-[10px] uppercase tracking-wider text-neutral-500">
            Contagion
          </div>
          <div className={cn("text-sm font-semibold", contagionConfig?.color)}>
            {contagionConfig?.label || "N/A"}
            {contagion && (
              <span className="ml-1 text-xs text-neutral-500 font-normal">
                ({(contagion.score * 100).toFixed(0)}%)
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Warnings */}
      {warnings.length > 0 && (
        <>
          <div className="w-px h-8 bg-neutral-700/50" />
          <Badge
            className={cn(
              "cursor-help",
              warnings.some((w) => w.includes("Crisis") || w.includes("Critical"))
                ? "bg-red-500/10 border-red-500/30 text-red-400"
                : "bg-amber-500/10 border-amber-500/30 text-amber-400"
            )}
            title={warnings.join("\n")}
          >
            <AlertTriangle className="w-3 h-3 mr-1" />
            {warnings.length} Warning{warnings.length > 1 ? "s" : ""}
          </Badge>
        </>
      )}

      {/* Refresh button */}
      <button
        onClick={() => refetch()}
        className="p-1 rounded hover:bg-neutral-700/50 transition-colors ml-auto"
        title="Refresh quantum status"
      >
        <RefreshCw className="w-3.5 h-3.5 text-neutral-500" />
      </button>
    </div>
  );
}

export default QuantumStatusWidget;
