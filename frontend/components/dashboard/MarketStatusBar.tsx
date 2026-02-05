"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import {
  Globe,
  Clock,
  Wifi,
  WifiOff,
  Activity,
  Radio,
} from "lucide-react";

interface MarketSession {
  name: string;
  shortName: string;
  isOpen: boolean;
  openTime: string;
  closeTime: string;
  timezone: string;
  flag: string;
}

interface MarketStatusBarProps {
  showFullDetails?: boolean;
}

function getCurrentTime(): string {
  return new Date().toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

function getMarketSessions(): MarketSession[] {
  const now = new Date();
  const utcHours = now.getUTCHours();
  const utcMinutes = now.getUTCMinutes();
  const totalMinutes = utcHours * 60 + utcMinutes;

  // Market hours in UTC
  const sessions: MarketSession[] = [
    {
      name: "New York",
      shortName: "NYSE",
      isOpen: totalMinutes >= 14 * 60 + 30 && totalMinutes < 21 * 60, // 14:30-21:00 UTC
      openTime: "09:30",
      closeTime: "16:00",
      timezone: "EST",
      flag: "🇺🇸",
    },
    {
      name: "London",
      shortName: "LSE",
      isOpen: totalMinutes >= 8 * 60 && totalMinutes < 16 * 60 + 30, // 08:00-16:30 UTC
      openTime: "08:00",
      closeTime: "16:30",
      timezone: "GMT",
      flag: "🇬🇧",
    },
    {
      name: "Tokyo",
      shortName: "TSE",
      isOpen: totalMinutes >= 0 && totalMinutes < 6 * 60, // 00:00-06:00 UTC
      openTime: "09:00",
      closeTime: "15:00",
      timezone: "JST",
      flag: "🇯🇵",
    },
    {
      name: "Hong Kong",
      shortName: "HKEX",
      isOpen: totalMinutes >= 1 * 60 + 30 && totalMinutes < 8 * 60, // 01:30-08:00 UTC
      openTime: "09:30",
      closeTime: "16:00",
      timezone: "HKT",
      flag: "🇭🇰",
    },
    {
      name: "Frankfurt",
      shortName: "FWB",
      isOpen: totalMinutes >= 8 * 60 && totalMinutes < 16 * 60 + 30, // 08:00-16:30 UTC
      openTime: "09:00",
      closeTime: "17:30",
      timezone: "CET",
      flag: "🇩🇪",
    },
  ];

  return sessions;
}

function SessionIndicator({
  session,
  compact = false,
}: {
  session: MarketSession;
  compact?: boolean;
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-2 px-2.5 py-1 rounded-md transition-all",
        session.isOpen
          ? "bg-green-500/10 border border-green-500/30"
          : "bg-neutral-800/50 border border-neutral-700/50"
      )}
    >
      <span className="text-sm">{session.flag}</span>
      <div className="flex flex-col">
        <span
          className={cn(
            "text-[10px] font-bold tracking-wide",
            session.isOpen ? "text-green-400" : "text-neutral-500"
          )}
        >
          {session.shortName}
        </span>
        {!compact && (
          <span className="text-[9px] text-neutral-600">
            {session.isOpen ? "OPEN" : "CLOSED"}
          </span>
        )}
      </div>
      <div
        className={cn(
          "w-1.5 h-1.5 rounded-full",
          session.isOpen ? "bg-green-500 animate-pulse" : "bg-neutral-600"
        )}
      />
    </div>
  );
}

function LiveClock() {
  const [time, setTime] = useState(getCurrentTime());

  useEffect(() => {
    const interval = setInterval(() => {
      setTime(getCurrentTime());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 bg-neutral-800/50 rounded-lg border border-neutral-700/50">
      <Clock className="w-4 h-4 text-blue-400" />
      <span className="font-mono text-lg font-bold text-neutral-100 tracking-wider">
        {time}
      </span>
      <span className="text-[10px] text-neutral-500 font-mono">UTC</span>
    </div>
  );
}

function ConnectionStatus() {
  const [isConnected, setIsConnected] = useState(true);
  const [latency, setLatency] = useState(12);

  // Simulate connection status
  useEffect(() => {
    const interval = setInterval(() => {
      setLatency(Math.floor(8 + Math.random() * 15));
      // Occasionally simulate disconnect
      if (Math.random() > 0.98) {
        setIsConnected(false);
        setTimeout(() => setIsConnected(true), 2000);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div
      className={cn(
        "flex items-center gap-2 px-2.5 py-1.5 rounded-lg border transition-all",
        isConnected
          ? "bg-green-500/5 border-green-500/20"
          : "bg-red-500/10 border-red-500/30"
      )}
    >
      {isConnected ? (
        <>
          <Wifi className="w-4 h-4 text-green-400" />
          <div className="flex flex-col">
            <span className="text-[10px] font-bold text-green-400">CONNECTED</span>
            <span className="text-[9px] text-neutral-500 font-mono">{latency}ms</span>
          </div>
        </>
      ) : (
        <>
          <WifiOff className="w-4 h-4 text-red-400 animate-pulse" />
          <span className="text-[10px] font-bold text-red-400">RECONNECTING...</span>
        </>
      )}
    </div>
  );
}

function DataFeedStatus() {
  const [feedRate, setFeedRate] = useState(1250);

  useEffect(() => {
    const interval = setInterval(() => {
      setFeedRate(1200 + Math.floor(Math.random() * 200));
    }, 500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-2 px-2.5 py-1.5 bg-purple-500/5 rounded-lg border border-purple-500/20">
      <Activity className="w-4 h-4 text-purple-400" />
      <div className="flex flex-col">
        <span className="text-[10px] font-bold text-purple-400">DATA FEED</span>
        <span className="text-[9px] text-neutral-500 font-mono">{feedRate}/s</span>
      </div>
    </div>
  );
}

export function MarketStatusBar({ showFullDetails = true }: MarketStatusBarProps) {
  const [sessions, setSessions] = useState<MarketSession[]>([]);

  useEffect(() => {
    setSessions(getMarketSessions());
    const interval = setInterval(() => {
      setSessions(getMarketSessions());
    }, 60000); // Update every minute
    return () => clearInterval(interval);
  }, []);

  const openCount = sessions.filter((s) => s.isOpen).length;

  return (
    <div className="flex items-center justify-between px-4 py-2 bg-neutral-900/80 border-b border-neutral-800 backdrop-blur-sm">
      {/* Left side - Market Sessions */}
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-2 pr-3 border-r border-neutral-800">
          <Globe className="w-4 h-4 text-cyan-400" />
          <span className="text-xs text-neutral-400">
            <span className="text-cyan-400 font-bold">{openCount}</span>/{sessions.length} Markets Open
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {sessions.map((session) => (
            <SessionIndicator key={session.shortName} session={session} compact={!showFullDetails} />
          ))}
        </div>
      </div>

      {/* Right side - Clock and Status */}
      <div className="flex items-center gap-3">
        {showFullDetails && (
          <>
            <DataFeedStatus />
            <ConnectionStatus />
          </>
        )}
        <LiveClock />

        {/* System status */}
        <div className="flex items-center gap-1.5 pl-3 border-l border-neutral-800">
          <Radio className="w-3.5 h-3.5 text-green-500 animate-pulse" />
          <span className="text-[10px] font-bold text-green-500 tracking-wide">
            SYSTEM ONLINE
          </span>
        </div>
      </div>
    </div>
  );
}
