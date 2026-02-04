"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import type { TargetLevel, TargetDirection } from "@/types/target-ladder";

interface TargetLadderProps {
  currentPrice: number;
  targets: TargetLevel[];
  direction: TargetDirection;
  symbol?: string;
  /** Animate steps on mount */
  animated?: boolean;
  /** Compact mode for smaller cards */
  compact?: boolean;
}

function formatPrice(price: number): string {
  if (price >= 10000) {
    return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  if (price >= 100) {
    return price.toFixed(2);
  }
  return price.toFixed(4);
}

function formatDelta(delta: number): string {
  const prefix = delta >= 0 ? "+" : "";
  if (Math.abs(delta) >= 100) {
    return `${prefix}${delta.toFixed(2)}`;
  }
  return `${prefix}${delta.toFixed(2)}`;
}

export function TargetLadder({
  currentPrice,
  targets,
  direction,
  symbol,
  animated = true,
  compact = false,
}: TargetLadderProps) {
  const [isVisible, setIsVisible] = useState(!animated);

  useEffect(() => {
    if (animated) {
      // Trigger animation after mount
      const timer = setTimeout(() => setIsVisible(true), 50);
      return () => clearTimeout(timer);
    }
  }, [animated]);

  // Sort targets by delta (ascending for bullish = closest first, descending for bearish)
  const sortedTargets = [...targets].sort((a, b) => {
    if (direction === "bullish") {
      return a.delta - b.delta;
    }
    return b.delta - a.delta;
  });

  // For bullish: show targets ascending from current price
  // For bearish: show targets descending from current price
  const displayTargets = direction === "bullish"
    ? sortedTargets.filter(t => t.delta > 0).reverse()
    : sortedTargets.filter(t => t.delta < 0);

  const isBullish = direction === "bullish";
  const directionColor = isBullish ? "text-green-500" : "text-red-500";
  const lineColor = isBullish ? "bg-green-500" : "bg-red-500";
  const stepBg = isBullish ? "bg-green-500/10" : "bg-red-500/10";
  const stepBorder = isBullish ? "border-green-500/30" : "border-red-500/30";

  return (
    <div className="relative">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs uppercase tracking-wider text-neutral-500">
          Target Ladder
        </span>
        <span className={cn("text-xs font-semibold uppercase", directionColor)}>
          {isBullish ? "↑ Bullish" : "↓ Bearish"}
        </span>
      </div>

      {/* Ladder visualization */}
      <div className="relative pl-4">
        {/* Vertical connector line */}
        <div
          className={cn(
            "absolute left-1 top-4 bottom-4 w-0.5 transition-all duration-500",
            lineColor
          )}
          style={{
            background: isBullish
              ? "linear-gradient(to top, rgb(34 197 94 / 0.3), rgb(34 197 94))"
              : "linear-gradient(to bottom, rgb(239 68 68 / 0.3), rgb(239 68 68))"
          }}
        />

        {/* Target steps */}
        <div className={cn("space-y-2", compact && "space-y-1.5")}>
          {displayTargets.map((target, index) => {
            const isFirst = index === 0;
            const animationDelay = index * 100;

            return (
              <div
                key={target.level}
                className={cn(
                  "relative flex items-center gap-3 rounded-lg border",
                  "transform transition-all duration-500 ease-out",
                  compact ? "p-1.5 gap-2" : "p-2",
                  stepBg,
                  stepBorder,
                  target.agreesWithDirection ? "opacity-100" : "opacity-50",
                  isFirst && "ring-1 ring-inset",
                  isFirst && (isBullish ? "ring-green-500/50" : "ring-red-500/50"),
                  // Animation states
                  isVisible
                    ? "translate-x-0 opacity-100"
                    : isBullish
                      ? "translate-y-4 opacity-0"
                      : "-translate-y-4 opacity-0"
                )}
                style={{
                  transitionDelay: `${animationDelay}ms`,
                }}
              >
                {/* Connection dot */}
                <div
                  className={cn(
                    "absolute -left-3.5 w-2 h-2 rounded-full transition-transform duration-300",
                    lineColor,
                    isFirst && "animate-pulse scale-125"
                  )}
                />

                {/* Level label */}
                <span className={cn(
                  "font-mono text-neutral-400",
                  compact ? "text-[10px] w-6" : "text-xs w-8"
                )}>
                  {target.level}
                </span>

                {/* Price */}
                <span className={cn(
                  "font-mono font-medium flex-1",
                  compact ? "text-xs" : "text-sm",
                  directionColor
                )}>
                  {formatPrice(target.price)}
                </span>

                {/* Delta */}
                <span className={cn(
                  "font-mono px-1.5 py-0.5 rounded",
                  compact ? "text-[10px]" : "text-xs",
                  isBullish ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"
                )}>
                  {formatDelta(target.delta)}
                </span>

                {/* Percentage delta - hidden in compact mode */}
                {!compact && (
                  <span className="font-mono text-xs text-neutral-500 w-14 text-right">
                    ({target.deltaPercent >= 0 ? "+" : ""}{target.deltaPercent.toFixed(2)}%)
                  </span>
                )}
              </div>
            );
          })}
        </div>

        {/* Current price marker */}
        <div className={cn(
          "relative border-t border-neutral-700 transition-all duration-500",
          compact ? "mt-2 pt-2" : "mt-3 pt-3",
          isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"
        )}
        style={{ transitionDelay: `${displayTargets.length * 100 + 100}ms` }}
        >
          <div
            className="absolute -left-3.5 top-1/2 -translate-y-1/2 w-3 h-3 rounded-full bg-blue-500 ring-2 ring-blue-500/30 animate-pulse"
          />
          <div className="flex items-center gap-3 pl-0">
            <span className={cn(
              "text-neutral-400",
              compact ? "text-[10px] w-6" : "text-xs w-8"
            )}>NOW</span>
            <span className={cn(
              "font-mono font-semibold text-blue-400",
              compact ? "text-xs" : "text-sm"
            )}>
              {symbol && <span className="text-neutral-500 mr-1">{symbol}</span>}
              {formatPrice(currentPrice)}
            </span>
            <span className={cn(compact ? "text-base" : "text-lg", directionColor)}>
              {isBullish ? "↑" : "↓"}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
