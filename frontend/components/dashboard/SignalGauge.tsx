"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

interface SignalGaugeProps {
  /** Signal strength from -100 (bearish) to +100 (bullish) */
  value: number;
  /** Confidence percentage 0-100 */
  confidence: number;
  /** Optional label */
  label?: string;
  /** Size variant */
  size?: "sm" | "md" | "lg";
  /** Enable animation */
  animated?: boolean;
  /** Show historical percentile ring */
  showPercentileRing?: boolean;
  /** Historical percentile (0-100) */
  percentile?: number;
}

function getNeedleColor(value: number): string {
  if (value > 50) return "#22c55e"; // green
  if (value > 20) return "#84cc16"; // lime
  if (value > -20) return "#f59e0b"; // amber
  if (value > -50) return "#f97316"; // orange
  return "#ef4444"; // red
}

function getSignalLabel(value: number): { text: string; color: string } {
  if (value > 70) return { text: "STRONG BUY", color: "text-green-400" };
  if (value > 30) return { text: "BUY", color: "text-green-500" };
  if (value > -30) return { text: "NEUTRAL", color: "text-amber-500" };
  if (value > -70) return { text: "SELL", color: "text-red-500" };
  return { text: "STRONG SELL", color: "text-red-400" };
}

export function SignalGauge({
  value,
  confidence,
  label,
  size = "md",
  animated = true,
  showPercentileRing = false,
  percentile = 50,
}: SignalGaugeProps) {
  const [animatedValue, setAnimatedValue] = useState(animated ? 0 : value);
  const [animatedConfidence, setAnimatedConfidence] = useState(animated ? 0 : confidence);
  const [isPulsing, setIsPulsing] = useState(false);

  useEffect(() => {
    if (!animated) return;

    // Animate value
    const valueTimer = setTimeout(() => setAnimatedValue(value), 100);
    const confTimer = setTimeout(() => setAnimatedConfidence(confidence), 200);

    // Pulse on strong signals
    if (Math.abs(value) > 60) {
      const pulseTimer = setTimeout(() => setIsPulsing(true), 1000);
      return () => {
        clearTimeout(valueTimer);
        clearTimeout(confTimer);
        clearTimeout(pulseTimer);
      };
    }

    return () => {
      clearTimeout(valueTimer);
      clearTimeout(confTimer);
    };
  }, [value, confidence, animated]);

  // Size configurations
  const sizes = {
    sm: { width: 160, height: 100, strokeWidth: 12, needleLength: 50 },
    md: { width: 220, height: 130, strokeWidth: 16, needleLength: 70 },
    lg: { width: 300, height: 170, strokeWidth: 20, needleLength: 95 },
  };

  const { width, height, strokeWidth, needleLength } = sizes[size];
  const centerX = width / 2;
  const centerY = height - 15;
  const radius = height - strokeWidth - 20;

  // Convert value (-100 to +100) to angle (-150 to -30 degrees, or 210 to 330 in standard)
  // We want -100 on left, 0 in middle, +100 on right
  const normalizedValue = (animatedValue + 100) / 200; // 0 to 1
  const needleAngle = -150 + normalizedValue * 120; // -150 to -30 degrees
  const needleRadians = (needleAngle * Math.PI) / 180;

  const needleX = centerX + needleLength * Math.cos(needleRadians);
  const needleY = centerY + needleLength * Math.sin(needleRadians);

  const signalInfo = getSignalLabel(animatedValue);
  const needleColor = getNeedleColor(animatedValue);

  // Create gradient arc segments
  const createArcPath = (startAngle: number, endAngle: number): string => {
    const startRad = (startAngle * Math.PI) / 180;
    const endRad = (endAngle * Math.PI) / 180;
    const x1 = centerX + radius * Math.cos(startRad);
    const y1 = centerY + radius * Math.sin(startRad);
    const x2 = centerX + radius * Math.cos(endRad);
    const y2 = centerY + radius * Math.sin(endRad);
    const largeArc = endAngle - startAngle > 180 ? 1 : 0;
    return `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`;
  };

  return (
    <div className="flex flex-col items-center">
      <svg
        width={width}
        height={height}
        className={cn(
          "transition-all duration-300",
          isPulsing && "drop-shadow-[0_0_15px_rgba(34,197,94,0.3)]"
        )}
      >
        <defs>
          {/* Gradient for the gauge arc */}
          <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#ef4444" />
            <stop offset="25%" stopColor="#f97316" />
            <stop offset="50%" stopColor="#f59e0b" />
            <stop offset="75%" stopColor="#84cc16" />
            <stop offset="100%" stopColor="#22c55e" />
          </linearGradient>

          {/* Glow filter */}
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Background arc */}
        <path
          d={createArcPath(-150, -30)}
          fill="none"
          stroke="#262626"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />

        {/* Colored arc segments */}
        <path
          d={createArcPath(-150, -126)}
          fill="none"
          stroke="#ef4444"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          opacity={0.8}
        />
        <path
          d={createArcPath(-126, -102)}
          fill="none"
          stroke="#f97316"
          strokeWidth={strokeWidth}
          strokeLinecap="butt"
          opacity={0.8}
        />
        <path
          d={createArcPath(-102, -78)}
          fill="none"
          stroke="#f59e0b"
          strokeWidth={strokeWidth}
          strokeLinecap="butt"
          opacity={0.8}
        />
        <path
          d={createArcPath(-78, -54)}
          fill="none"
          stroke="#84cc16"
          strokeWidth={strokeWidth}
          strokeLinecap="butt"
          opacity={0.8}
        />
        <path
          d={createArcPath(-54, -30)}
          fill="none"
          stroke="#22c55e"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          opacity={0.8}
        />

        {/* Tick marks */}
        {[-100, -50, 0, 50, 100].map((tick) => {
          const tickNorm = (tick + 100) / 200;
          const tickAngle = -150 + tickNorm * 120;
          const tickRad = (tickAngle * Math.PI) / 180;
          const innerRadius = radius - strokeWidth / 2 - 8;
          const outerRadius = radius - strokeWidth / 2 - 2;
          const x1 = centerX + innerRadius * Math.cos(tickRad);
          const y1 = centerY + innerRadius * Math.sin(tickRad);
          const x2 = centerX + outerRadius * Math.cos(tickRad);
          const y2 = centerY + outerRadius * Math.sin(tickRad);

          return (
            <line
              key={tick}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="#525252"
              strokeWidth={2}
            />
          );
        })}

        {/* Percentile ring (optional) */}
        {showPercentileRing && (
          <>
            <circle
              cx={centerX}
              cy={centerY}
              r={radius + strokeWidth / 2 + 6}
              fill="none"
              stroke="#1f1f1f"
              strokeWidth={4}
            />
            <path
              d={createArcPath(-150, -150 + (percentile / 100) * 120)}
              fill="none"
              stroke="#6366f1"
              strokeWidth={4}
              strokeLinecap="round"
              style={{
                transform: `translate(0, 0)`,
                transformOrigin: `${centerX}px ${centerY}px`,
              }}
            />
          </>
        )}

        {/* Needle */}
        <g
          className="transition-transform duration-1000 ease-out"
          style={{
            transformOrigin: `${centerX}px ${centerY}px`,
          }}
        >
          {/* Needle shadow */}
          <line
            x1={centerX}
            y1={centerY}
            x2={needleX}
            y2={needleY}
            stroke="rgba(0,0,0,0.3)"
            strokeWidth={4}
            strokeLinecap="round"
            transform="translate(2, 2)"
          />

          {/* Needle body */}
          <line
            x1={centerX}
            y1={centerY}
            x2={needleX}
            y2={needleY}
            stroke={needleColor}
            strokeWidth={4}
            strokeLinecap="round"
            filter="url(#glow)"
            className="transition-all duration-1000 ease-out"
          />

          {/* Needle center cap */}
          <circle
            cx={centerX}
            cy={centerY}
            r={8}
            fill="#171717"
            stroke={needleColor}
            strokeWidth={3}
          />
        </g>

        {/* Value display */}
        <text
          x={centerX}
          y={centerY - 25}
          textAnchor="middle"
          className="fill-neutral-100 font-mono font-bold"
          style={{ fontSize: size === "lg" ? 28 : size === "md" ? 22 : 16 }}
        >
          {animatedValue > 0 ? "+" : ""}{animatedValue.toFixed(0)}
        </text>

        {/* Labels */}
        <text
          x={20}
          y={centerY + 5}
          textAnchor="start"
          className="fill-red-500 font-mono text-xs"
        >
          SELL
        </text>
        <text
          x={width - 20}
          y={centerY + 5}
          textAnchor="end"
          className="fill-green-500 font-mono text-xs"
        >
          BUY
        </text>
      </svg>

      {/* Signal label and confidence */}
      <div className="text-center mt-2 space-y-1">
        <div className={cn("text-sm font-bold tracking-wider", signalInfo.color)}>
          {signalInfo.text}
        </div>
        <div className="flex items-center justify-center gap-2">
          <span className="text-xs text-neutral-500">Confidence:</span>
          <div className="flex items-center gap-1.5">
            <div className="w-16 h-1.5 bg-neutral-800 rounded-full overflow-hidden">
              <div
                className={cn(
                  "h-full rounded-full transition-all duration-700",
                  animatedConfidence >= 70 ? "bg-green-500" :
                  animatedConfidence >= 50 ? "bg-blue-500" :
                  animatedConfidence >= 30 ? "bg-yellow-500" : "bg-red-500"
                )}
                style={{ width: `${animatedConfidence}%` }}
              />
            </div>
            <span className="text-xs font-mono text-neutral-400">
              {animatedConfidence.toFixed(0)}%
            </span>
          </div>
        </div>
        {label && (
          <div className="text-xs text-neutral-600 mt-1">{label}</div>
        )}
      </div>
    </div>
  );
}
