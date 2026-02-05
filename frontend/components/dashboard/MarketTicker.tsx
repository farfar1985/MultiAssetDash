"use client";

import { useEffect, useState, useRef } from "react";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface TickerItem {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume?: string;
}

interface MarketTickerProps {
  items?: TickerItem[];
  speed?: "slow" | "normal" | "fast";
  showVolume?: boolean;
  pauseOnHover?: boolean;
}

const DEFAULT_TICKER_DATA: TickerItem[] = [
  { symbol: "CL", name: "Crude Oil", price: 73.52, change: 0.84, changePercent: 1.15, volume: "1.2M" },
  { symbol: "GC", name: "Gold", price: 2051.30, change: -8.20, changePercent: -0.40, volume: "892K" },
  { symbol: "BTC", name: "Bitcoin", price: 45234.50, change: 1250.00, changePercent: 2.84, volume: "32.4B" },
  { symbol: "ES", name: "S&P 500 Fut", price: 4892.25, change: 12.50, changePercent: 0.26, volume: "2.1M" },
  { symbol: "NQ", name: "Nasdaq Fut", price: 17425.00, change: 85.75, changePercent: 0.49, volume: "1.8M" },
  { symbol: "NG", name: "Natural Gas", price: 2.634, change: -0.082, changePercent: -3.02, volume: "456K" },
  { symbol: "SI", name: "Silver", price: 23.45, change: 0.32, changePercent: 1.38, volume: "234K" },
  { symbol: "EUR", name: "EUR/USD", price: 1.0892, change: 0.0023, changePercent: 0.21, volume: "8.2B" },
  { symbol: "VIX", name: "Volatility", price: 14.52, change: -0.84, changePercent: -5.47, volume: "342K" },
  { symbol: "DXY", name: "Dollar Index", price: 103.42, change: -0.18, changePercent: -0.17, volume: "1.1M" },
  { symbol: "10Y", name: "10Y Treasury", price: 4.285, change: 0.032, changePercent: 0.75, volume: "5.4B" },
  { symbol: "ETH", name: "Ethereum", price: 2485.30, change: 78.50, changePercent: 3.26, volume: "14.2B" },
];

function TickerItemDisplay({
  item,
  showVolume,
}: {
  item: TickerItem;
  showVolume: boolean;
}) {
  const isPositive = item.change > 0;
  const isNeutral = item.change === 0;

  return (
    <div className="flex items-center gap-3 px-4 py-1 border-r border-neutral-800/50 whitespace-nowrap">
      {/* Symbol */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-mono font-bold text-neutral-300">
          {item.symbol}
        </span>
        <span className="text-[10px] text-neutral-600 hidden lg:inline">
          {item.name}
        </span>
      </div>

      {/* Price */}
      <span className="font-mono text-sm text-neutral-100">
        {item.price.toLocaleString(undefined, {
          minimumFractionDigits: item.price < 10 ? 3 : 2,
          maximumFractionDigits: item.price < 10 ? 3 : 2,
        })}
      </span>

      {/* Change */}
      <div
        className={cn(
          "flex items-center gap-1",
          isPositive ? "text-green-400" : isNeutral ? "text-neutral-500" : "text-red-400"
        )}
      >
        {isPositive ? (
          <TrendingUp className="w-3 h-3" />
        ) : isNeutral ? (
          <Minus className="w-3 h-3" />
        ) : (
          <TrendingDown className="w-3 h-3" />
        )}
        <span className="font-mono text-xs">
          {isPositive ? "+" : ""}
          {item.changePercent.toFixed(2)}%
        </span>
      </div>

      {/* Volume (optional) */}
      {showVolume && item.volume && (
        <span className="text-[10px] text-neutral-600 font-mono">
          Vol: {item.volume}
        </span>
      )}
    </div>
  );
}

export function MarketTicker({
  items = DEFAULT_TICKER_DATA,
  speed = "normal",
  showVolume = false,
  pauseOnHover = true,
}: MarketTickerProps) {
  const [isPaused, setIsPaused] = useState(false);
  const [tickerItems, setTickerItems] = useState(items);
  const containerRef = useRef<HTMLDivElement>(null);

  // Simulate real-time price updates
  useEffect(() => {
    const interval = setInterval(() => {
      setTickerItems((prev) =>
        prev.map((item) => {
          // Random price fluctuation
          const fluctuation = (Math.random() - 0.5) * 0.002 * item.price;
          const newPrice = item.price + fluctuation;
          const newChange = item.change + fluctuation;

          return {
            ...item,
            price: newPrice,
            change: newChange,
            changePercent: (newChange / (newPrice - newChange)) * 100,
          };
        })
      );
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const speedDuration = {
    slow: "60s",
    normal: "40s",
    fast: "25s",
  };

  // Triple the items for seamless loop
  const displayItems = [...tickerItems, ...tickerItems, ...tickerItems];

  return (
    <div
      ref={containerRef}
      className="relative overflow-hidden bg-neutral-950 border-b border-neutral-800"
      onMouseEnter={() => pauseOnHover && setIsPaused(true)}
      onMouseLeave={() => pauseOnHover && setIsPaused(false)}
    >
      {/* Gradient overlays for fade effect */}
      <div className="absolute left-0 top-0 bottom-0 w-16 bg-gradient-to-r from-neutral-950 to-transparent z-10 pointer-events-none" />
      <div className="absolute right-0 top-0 bottom-0 w-16 bg-gradient-to-l from-neutral-950 to-transparent z-10 pointer-events-none" />

      {/* Live indicator */}
      <div className="absolute left-4 top-1/2 -translate-y-1/2 z-20 flex items-center gap-1.5">
        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
        <span className="text-[10px] font-bold text-green-500 tracking-wider">LIVE</span>
      </div>

      {/* Scrolling content */}
      <div
        className={cn(
          "flex items-center py-1.5 pl-20",
          !isPaused && "animate-ticker"
        )}
        style={{
          animationDuration: speedDuration[speed],
          animationPlayState: isPaused ? "paused" : "running",
        }}
      >
        {displayItems.map((item, index) => (
          <TickerItemDisplay
            key={`${item.symbol}-${index}`}
            item={item}
            showVolume={showVolume}
          />
        ))}
      </div>

      {/* Pause indicator */}
      {isPaused && (
        <div className="absolute right-20 top-1/2 -translate-y-1/2 z-20">
          <span className="text-[10px] text-amber-500 font-mono">PAUSED</span>
        </div>
      )}

      <style jsx>{`
        @keyframes ticker {
          0% {
            transform: translateX(0);
          }
          100% {
            transform: translateX(-33.333%);
          }
        }
        .animate-ticker {
          animation: ticker linear infinite;
        }
      `}</style>
    </div>
  );
}
