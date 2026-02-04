"use client";

import { useBackendHealth } from "@/hooks/useApi";
import { cn } from "@/lib/utils";
import { Activity, AlertCircle, CheckCircle, Loader2, RefreshCw } from "lucide-react";

type ConnectionStatus = "connected" | "disconnected" | "connecting" | "error";

interface ApiHealthIndicatorProps {
  className?: string;
  showDetails?: boolean;
}

export function ApiHealthIndicator({ className, showDetails = false }: ApiHealthIndicatorProps) {
  const { data, isLoading, isError, error, refetch, dataUpdatedAt } = useBackendHealth();

  const getStatus = (): ConnectionStatus => {
    if (isLoading) return "connecting";
    if (isError) return "error";
    if (data?.success) return "connected";
    return "disconnected";
  };

  const status = getStatus();

  const statusConfig = {
    connected: {
      icon: CheckCircle,
      color: "text-green-500",
      bg: "bg-green-500/10",
      border: "border-green-500/30",
      dot: "bg-green-500",
      label: "API Connected",
      description: data?.service ? `${data.service} v${data.version}` : "Backend online",
    },
    disconnected: {
      icon: AlertCircle,
      color: "text-yellow-500",
      bg: "bg-yellow-500/10",
      border: "border-yellow-500/30",
      dot: "bg-yellow-500",
      label: "Disconnected",
      description: "Unable to reach backend",
    },
    connecting: {
      icon: Loader2,
      color: "text-blue-500",
      bg: "bg-blue-500/10",
      border: "border-blue-500/30",
      dot: "bg-blue-500",
      label: "Connecting",
      description: "Checking backend status...",
    },
    error: {
      icon: AlertCircle,
      color: "text-red-500",
      bg: "bg-red-500/10",
      border: "border-red-500/30",
      dot: "bg-red-500",
      label: "Connection Error",
      description: error?.message || "Failed to connect",
    },
  };

  const config = statusConfig[status];
  const Icon = config.icon;

  const lastChecked = dataUpdatedAt
    ? new Date(dataUpdatedAt).toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      })
    : null;

  if (!showDetails) {
    // Compact version for footer
    return (
      <div
        className={cn(
          "flex items-center gap-2 px-3 py-1.5 rounded-lg border transition-colors cursor-pointer hover:opacity-80",
          config.bg,
          config.border,
          className
        )}
        onClick={() => refetch()}
        title={`${config.label}: ${config.description}. Click to refresh.`}
      >
        <div className={cn("w-2 h-2 rounded-full", config.dot, status === "connecting" && "animate-pulse")} />
        <Icon className={cn("w-3.5 h-3.5", config.color, status === "connecting" && "animate-spin")} />
        <span className={cn("text-xs font-medium", config.color)}>{config.label}</span>
      </div>
    );
  }

  // Detailed version
  return (
    <div
      className={cn(
        "flex items-center gap-3 px-4 py-2.5 rounded-lg border",
        config.bg,
        config.border,
        className
      )}
    >
      <div className="flex items-center gap-2">
        <div className={cn("w-2.5 h-2.5 rounded-full", config.dot, status === "connecting" && "animate-pulse")} />
        <Icon className={cn("w-4 h-4", config.color, status === "connecting" && "animate-spin")} />
      </div>

      <div className="flex-1 min-w-0">
        <div className={cn("text-sm font-medium", config.color)}>{config.label}</div>
        <div className="text-xs text-neutral-500 truncate">{config.description}</div>
      </div>

      <div className="flex items-center gap-2">
        {lastChecked && (
          <span className="text-[10px] text-neutral-600 font-mono">
            Last: {lastChecked}
          </span>
        )}
        <button
          onClick={() => refetch()}
          className={cn(
            "p-1 rounded hover:bg-neutral-800 transition-colors",
            isLoading && "cursor-not-allowed opacity-50"
          )}
          disabled={isLoading}
          title="Refresh connection status"
        >
          <RefreshCw className={cn("w-3.5 h-3.5 text-neutral-400", isLoading && "animate-spin")} />
        </button>
      </div>
    </div>
  );
}

/**
 * Footer component with API health status
 */
export function DashboardFooter() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";

  return (
    <footer className="flex items-center justify-between px-4 py-2 border-t border-neutral-800 bg-neutral-950/80">
      <div className="flex items-center gap-4">
        <ApiHealthIndicator />
        <span className="text-[10px] text-neutral-600 font-mono hidden sm:inline">
          {apiUrl}
        </span>
      </div>

      <div className="flex items-center gap-4 text-[10px] text-neutral-600">
        <span className="flex items-center gap-1.5">
          <Activity className="w-3 h-3" />
          <span className="font-mono">QDT Nexus</span>
        </span>
        <span className="hidden md:inline">Powered by Quantum Decision Theory</span>
      </div>
    </footer>
  );
}

export default ApiHealthIndicator;
