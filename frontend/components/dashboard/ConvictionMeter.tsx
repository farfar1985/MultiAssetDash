"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import type { ConvictionLevel } from "@/types/target-ladder";

interface ConvictionMeterProps {
  consensus: number;
  targetsAgreeing: number;
  targetsTotal: number;
  conviction: ConvictionLevel;
  variant?: "circular" | "bar";
  /** Label type: "targets" or "horizons" */
  labelType?: "targets" | "horizons";
  /** Animate on mount */
  animated?: boolean;
  /** Size variant for circular */
  size?: "sm" | "md" | "lg";
}

function getConvictionColor(conviction: ConvictionLevel): {
  bg: string;
  text: string;
  ring: string;
  fill: string;
} {
  switch (conviction) {
    case "HIGH":
      return {
        bg: "bg-green-500/10",
        text: "text-green-500",
        ring: "ring-green-500/30",
        fill: "stroke-green-500",
      };
    case "MEDIUM":
      return {
        bg: "bg-yellow-500/10",
        text: "text-yellow-500",
        ring: "ring-yellow-500/30",
        fill: "stroke-yellow-500",
      };
    case "LOW":
      return {
        bg: "bg-red-500/10",
        text: "text-red-500",
        ring: "ring-red-500/30",
        fill: "stroke-red-500",
      };
  }
}

function CircularMeter({
  consensus,
  conviction,
  targetsAgreeing,
  targetsTotal,
  labelType = "targets",
  animated = true,
  size: sizeVariant = "md",
}: ConvictionMeterProps) {
  const [animatedConsensus, setAnimatedConsensus] = useState(animated ? 0 : consensus);
  const colors = getConvictionColor(conviction);

  // Animate consensus value on mount
  useEffect(() => {
    if (!animated) return;
    const timer = setTimeout(() => setAnimatedConsensus(consensus), 100);
    return () => clearTimeout(timer);
  }, [consensus, animated]);

  // Size variants
  const sizes = {
    sm: { size: 64, strokeWidth: 5, fontSize: "text-base" },
    md: { size: 80, strokeWidth: 6, fontSize: "text-lg" },
    lg: { size: 100, strokeWidth: 7, fontSize: "text-xl" },
  };

  const { size, strokeWidth, fontSize } = sizes[sizeVariant];
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (animatedConsensus / 100) * circumference;

  const labelText = labelType === "horizons" ? "horizons agree" : "targets agree";

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative">
        <svg width={size} height={size} className="-rotate-90">
          {/* Background circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth={strokeWidth}
            className="text-neutral-800"
          />
          {/* Progress circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className={cn(colors.fill, "transition-all duration-1000 ease-out")}
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={cn("font-mono font-bold", fontSize, colors.text)}>
            {animatedConsensus.toFixed(animatedConsensus % 1 !== 0 ? 2 : 0)}%
          </span>
        </div>
      </div>

      {/* Label */}
      <div className="text-center">
        <div className={cn("text-xs font-semibold uppercase tracking-wider", colors.text)}>
          {conviction}
        </div>
        <div className="text-xs text-neutral-500 font-mono mt-0.5">
          {targetsAgreeing}/{targetsTotal} {labelText}
        </div>
      </div>
    </div>
  );
}

function BarMeter({
  consensus,
  conviction,
  targetsAgreeing,
  targetsTotal,
  labelType = "targets",
  animated = true,
}: ConvictionMeterProps) {
  const [animatedWidth, setAnimatedWidth] = useState(animated ? 0 : consensus);
  const [showIndicators, setShowIndicators] = useState(!animated);
  const colors = getConvictionColor(conviction);

  useEffect(() => {
    if (!animated) return;
    const timer1 = setTimeout(() => setAnimatedWidth(consensus), 100);
    const timer2 = setTimeout(() => setShowIndicators(true), 300);
    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
    };
  }, [consensus, animated]);

  const labelText = labelType === "horizons" ? "agree" : "agree";

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs uppercase tracking-wider text-neutral-500">
          Conviction
        </span>
        <span className={cn("text-xs font-semibold", colors.text)}>
          {conviction}
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-2.5 bg-neutral-800 rounded-full overflow-hidden">
        <div
          className={cn(
            "h-full rounded-full transition-all duration-700 ease-out",
            conviction === "HIGH" && "bg-green-500",
            conviction === "MEDIUM" && "bg-yellow-500",
            conviction === "LOW" && "bg-red-500"
          )}
          style={{ width: `${animatedWidth}%` }}
        />
      </div>

      {/* Stats row */}
      <div className="flex items-center justify-between">
        <span className={cn("font-mono text-sm font-semibold", colors.text)}>
          {consensus.toFixed(consensus % 1 !== 0 ? 2 : 0)}% consensus
        </span>
        <span className="text-xs text-neutral-500 font-mono">
          {targetsAgreeing}/{targetsTotal} {labelText}
        </span>
      </div>

      {/* Visual target/horizon indicators */}
      <div className="flex items-center gap-1 justify-center pt-1">
        {Array.from({ length: targetsTotal }).map((_, i) => (
          <div
            key={i}
            className={cn(
              "w-3 h-3 rounded-sm transition-all duration-300",
              showIndicators && i < targetsAgreeing
                ? cn(
                    conviction === "HIGH" && "bg-green-500",
                    conviction === "MEDIUM" && "bg-yellow-500",
                    conviction === "LOW" && "bg-red-500"
                  )
                : "bg-neutral-700"
            )}
            style={{
              transitionDelay: `${i * 50}ms`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

export function ConvictionMeter({
  consensus,
  targetsAgreeing,
  targetsTotal,
  conviction,
  variant = "bar",
  labelType = "targets",
  animated = true,
  size = "md",
}: ConvictionMeterProps) {
  if (variant === "circular") {
    return (
      <CircularMeter
        consensus={consensus}
        targetsAgreeing={targetsAgreeing}
        targetsTotal={targetsTotal}
        conviction={conviction}
        labelType={labelType}
        animated={animated}
        size={size}
      />
    );
  }

  return (
    <BarMeter
      consensus={consensus}
      targetsAgreeing={targetsAgreeing}
      targetsTotal={targetsTotal}
      conviction={conviction}
      labelType={labelType}
      animated={animated}
    />
  );
}
