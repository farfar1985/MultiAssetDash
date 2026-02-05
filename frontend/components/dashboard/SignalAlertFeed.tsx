"use client";

import { useState, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Bell,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Zap,
  Target,
  XCircle,
  Clock,
  Volume2,
  VolumeX,
} from "lucide-react";

type AlertType = "signal_new" | "signal_update" | "target_hit" | "stop_hit" | "volatility" | "consensus_shift";
type AlertPriority = "high" | "medium" | "low";

interface AlertItem {
  id: string;
  type: AlertType;
  priority: AlertPriority;
  asset: string;
  symbol: string;
  message: string;
  detail?: string;
  timestamp: Date;
  isNew?: boolean;
}

interface SignalAlertFeedProps {
  maxAlerts?: number;
  autoPlay?: boolean;
  enableSound?: boolean;
  compact?: boolean;
}

const ALERT_ICONS: Record<AlertType, React.ElementType> = {
  signal_new: Zap,
  signal_update: TrendingUp,
  target_hit: Target,
  stop_hit: XCircle,
  volatility: AlertTriangle,
  consensus_shift: TrendingDown,
};

const ALERT_COLORS: Record<AlertType, { bg: string; border: string; text: string }> = {
  signal_new: { bg: "bg-purple-500/10", border: "border-purple-500/30", text: "text-purple-400" },
  signal_update: { bg: "bg-blue-500/10", border: "border-blue-500/30", text: "text-blue-400" },
  target_hit: { bg: "bg-green-500/10", border: "border-green-500/30", text: "text-green-400" },
  stop_hit: { bg: "bg-red-500/10", border: "border-red-500/30", text: "text-red-400" },
  volatility: { bg: "bg-amber-500/10", border: "border-amber-500/30", text: "text-amber-400" },
  consensus_shift: { bg: "bg-cyan-500/10", border: "border-cyan-500/30", text: "text-cyan-400" },
};

const PRIORITY_PULSE: Record<AlertPriority, string> = {
  high: "animate-pulse",
  medium: "",
  low: "",
};

function generateMockAlert(): AlertItem {
  const types: AlertType[] = ["signal_new", "signal_update", "target_hit", "stop_hit", "volatility", "consensus_shift"];
  const assets = [
    { name: "Crude Oil", symbol: "CL" },
    { name: "Gold", symbol: "GC" },
    { name: "Bitcoin", symbol: "BTC" },
    { name: "S&P 500", symbol: "SPY" },
    { name: "Natural Gas", symbol: "NG" },
    { name: "EUR/USD", symbol: "EUR" },
  ];

  const type = types[Math.floor(Math.random() * types.length)];
  const asset = assets[Math.floor(Math.random() * assets.length)];

  const messages: Record<AlertType, string[]> = {
    signal_new: ["New BULLISH signal generated", "New BEARISH signal generated", "Fresh signal detected"],
    signal_update: ["Signal strength increased to 85", "Confidence level upgraded", "Signal confirmed by ensemble"],
    target_hit: ["T1 target achieved (+1.5%)", "T2 target reached (+2.8%)", "Price target hit"],
    stop_hit: ["Stop loss triggered (-1.2%)", "Position closed at stop", "Risk limit reached"],
    volatility: ["Unusual volatility detected", "Volume spike observed", "Breaking out of range"],
    consensus_shift: ["Model consensus shifting bearish", "Bullish momentum building", "Sentiment reversal detected"],
  };

  const priority: AlertPriority =
    type === "signal_new" || type === "stop_hit" ? "high" :
    type === "target_hit" || type === "volatility" ? "medium" : "low";

  return {
    id: `alert-${Date.now()}-${Math.random()}`,
    type,
    priority,
    asset: asset.name,
    symbol: asset.symbol,
    message: messages[type][Math.floor(Math.random() * messages[type].length)],
    detail: `Signal strength: ${(60 + Math.random() * 35).toFixed(0)}%`,
    timestamp: new Date(),
    isNew: true,
  };
}

function formatTime(date: Date): string {
  return date.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function AlertCard({
  alert,
  compact,
  onDismiss,
}: {
  alert: AlertItem;
  compact: boolean;
  onDismiss?: (id: string) => void;
}) {
  const Icon = ALERT_ICONS[alert.type];
  const colors = ALERT_COLORS[alert.type];
  const pulseClass = alert.isNew ? PRIORITY_PULSE[alert.priority] : "";

  return (
    <div
      className={cn(
        "flex items-start gap-3 p-3 rounded-lg border transition-all duration-500",
        colors.bg,
        colors.border,
        alert.isNew && "ring-1 ring-white/10",
        pulseClass
      )}
    >
      {/* Icon */}
      <div className={cn("p-1.5 rounded-md", colors.bg)}>
        <Icon className={cn("w-4 h-4", colors.text)} />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-xs font-mono font-bold text-neutral-200">
            {alert.symbol}
          </span>
          {!compact && (
            <span className="text-[10px] text-neutral-500 truncate">
              {alert.asset}
            </span>
          )}
          {alert.priority === "high" && (
            <Badge className="bg-red-500/20 border-red-500/30 text-red-400 text-[9px] px-1 py-0">
              URGENT
            </Badge>
          )}
        </div>

        <p className={cn("text-sm font-medium", colors.text)}>
          {alert.message}
        </p>

        {!compact && alert.detail && (
          <p className="text-xs text-neutral-500 mt-1">{alert.detail}</p>
        )}

        <div className="flex items-center gap-1.5 mt-1.5">
          <Clock className="w-3 h-3 text-neutral-600" />
          <span className="text-[10px] font-mono text-neutral-600">
            {formatTime(alert.timestamp)}
          </span>
        </div>
      </div>

      {/* Dismiss */}
      {onDismiss && (
        <button
          onClick={() => onDismiss(alert.id)}
          className="text-neutral-600 hover:text-neutral-400 transition-colors"
        >
          <XCircle className="w-4 h-4" />
        </button>
      )}
    </div>
  );
}

export function SignalAlertFeed({
  maxAlerts = 8,
  autoPlay = true,
  enableSound = false,
  compact = false,
}: SignalAlertFeedProps) {
  const [alerts, setAlerts] = useState<AlertItem[]>([]);
  const [isSoundEnabled, setIsSoundEnabled] = useState(enableSound);
  const [isPaused, setIsPaused] = useState(!autoPlay);
  const feedRef = useRef<HTMLDivElement>(null);

  // Auto-generate alerts
  useEffect(() => {
    if (isPaused) return;

    // Initial alerts
    const initialAlerts = Array.from({ length: 4 }, () => ({
      ...generateMockAlert(),
      isNew: false,
    }));
    setAlerts(initialAlerts);

    // Add new alerts periodically
    const interval = setInterval(() => {
      setAlerts((prev) => {
        const newAlert = generateMockAlert();
        const updated = [newAlert, ...prev.map((a) => ({ ...a, isNew: false }))];

        // Trim to max
        if (updated.length > maxAlerts) {
          return updated.slice(0, maxAlerts);
        }
        return updated;
      });

      // Play sound for high priority
      if (isSoundEnabled) {
        // Would play sound here
      }
    }, 8000 + Math.random() * 7000); // Random 8-15 seconds

    return () => clearInterval(interval);
  }, [isPaused, maxAlerts, isSoundEnabled]);

  const handleDismiss = (id: string) => {
    setAlerts((prev) => prev.filter((a) => a.id !== id));
  };

  const handleClearAll = () => {
    setAlerts([]);
  };

  const unreadCount = alerts.filter((a) => a.isNew).length;

  return (
    <div className="flex flex-col h-full bg-neutral-900/50 border border-neutral-800 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-neutral-800 bg-neutral-900/80">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Bell className="w-5 h-5 text-purple-400" />
            {unreadCount > 0 && (
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full flex items-center justify-center">
                <span className="text-[9px] font-bold text-white">{unreadCount}</span>
              </div>
            )}
          </div>
          <div>
            <h3 className="text-sm font-semibold text-neutral-200">Signal Alerts</h3>
            <span className="text-[10px] text-neutral-500">Real-time notifications</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Sound toggle */}
          <button
            onClick={() => setIsSoundEnabled(!isSoundEnabled)}
            className={cn(
              "p-1.5 rounded-md transition-colors",
              isSoundEnabled
                ? "bg-purple-500/20 text-purple-400"
                : "bg-neutral-800 text-neutral-500 hover:text-neutral-400"
            )}
          >
            {isSoundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>

          {/* Pause toggle */}
          <button
            onClick={() => setIsPaused(!isPaused)}
            className={cn(
              "px-2 py-1 text-[10px] font-bold rounded-md transition-colors",
              isPaused
                ? "bg-amber-500/20 text-amber-400"
                : "bg-green-500/20 text-green-400"
            )}
          >
            {isPaused ? "PAUSED" : "LIVE"}
          </button>

          {/* Clear all */}
          {alerts.length > 0 && (
            <button
              onClick={handleClearAll}
              className="text-[10px] text-neutral-500 hover:text-neutral-300 transition-colors"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Alert Feed */}
      <div
        ref={feedRef}
        className="flex-1 overflow-y-auto p-3 space-y-2 scrollbar-thin scrollbar-track-neutral-900 scrollbar-thumb-neutral-700"
      >
        {alerts.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full py-8 text-center">
            <Bell className="w-8 h-8 text-neutral-700 mb-2" />
            <p className="text-sm text-neutral-500">No alerts yet</p>
            <p className="text-xs text-neutral-600">New signals will appear here</p>
          </div>
        ) : (
          alerts.map((alert) => (
            <AlertCard
              key={alert.id}
              alert={alert}
              compact={compact}
              onDismiss={handleDismiss}
            />
          ))
        )}
      </div>

      {/* Footer status */}
      <div className="px-4 py-2 border-t border-neutral-800 bg-neutral-900/80">
        <div className="flex items-center justify-between text-[10px]">
          <span className="text-neutral-600">
            {alerts.length} alerts • {alerts.filter((a) => a.priority === "high").length} urgent
          </span>
          <div className="flex items-center gap-1.5">
            <div className={cn(
              "w-1.5 h-1.5 rounded-full",
              isPaused ? "bg-amber-500" : "bg-green-500 animate-pulse"
            )} />
            <span className={cn(
              "font-mono",
              isPaused ? "text-amber-500" : "text-green-500"
            )}>
              {isPaused ? "Feed paused" : "Monitoring active"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
