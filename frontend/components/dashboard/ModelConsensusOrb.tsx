"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Zap,
  Activity,
  Users,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface ModelVotes {
  bullish: number;
  bearish: number;
  neutral: number;
  totalModels: number;
  consensusStrength: number; // 0-100, how aligned are the models
  dominantSignal: "bullish" | "bearish" | "neutral";
  lastUpdated: Date;
}

interface Particle {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  color: string;
  opacity: number;
  life: number;
  maxLife: number;
}

interface ModelConsensusOrbProps {
  votes: ModelVotes;
  size?: "sm" | "md" | "lg";
  showDetails?: boolean;
  animated?: boolean;
  onSignalClick?: (signal: "bullish" | "bearish" | "neutral") => void;
}

// ============================================================================
// Color Utilities
// ============================================================================

function getConsensusColor(signal: "bullish" | "bearish" | "neutral"): string {
  switch (signal) {
    case "bullish":
      return "#22c55e";
    case "bearish":
      return "#ef4444";
    case "neutral":
      return "#f59e0b";
  }
}

function getGlowColor(signal: "bullish" | "bearish" | "neutral", opacity = 0.5): string {
  switch (signal) {
    case "bullish":
      return `rgba(34, 197, 94, ${opacity})`;
    case "bearish":
      return `rgba(239, 68, 68, ${opacity})`;
    case "neutral":
      return `rgba(245, 158, 11, ${opacity})`;
  }
}

// ============================================================================
// Particle System Hook
// ============================================================================

function useParticleSystem(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  votes: ModelVotes,
  enabled: boolean
) {
  const particlesRef = useRef<Particle[]>([]);
  const animationRef = useRef<number | null>(null);
  const lastSpawnRef = useRef(0);

  const spawnParticle = useCallback((width: number, height: number): Particle => {
    const centerX = width / 2;
    const centerY = height / 2;

    // Spawn particles from the edge
    const angle = Math.random() * Math.PI * 2;
    const radius = Math.max(width, height) * 0.5;
    const x = centerX + Math.cos(angle) * radius;
    const y = centerY + Math.sin(angle) * radius;

    // Determine particle color based on vote distribution
    const rand = Math.random() * votes.totalModels;
    let color: string;
    if (rand < votes.bullish) {
      color = "#22c55e";
    } else if (rand < votes.bullish + votes.bearish) {
      color = "#ef4444";
    } else {
      color = "#f59e0b";
    }

    // Velocity toward center
    const speed = 0.5 + Math.random() * 1.5;
    const vx = (centerX - x) / radius * speed;
    const vy = (centerY - y) / radius * speed;

    return {
      id: Math.random(),
      x,
      y,
      vx,
      vy,
      size: 1 + Math.random() * 2,
      color,
      opacity: 0.6 + Math.random() * 0.4,
      life: 0,
      maxLife: 60 + Math.random() * 40,
    };
  }, [votes]);

  useEffect(() => {
    if (!enabled) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const coreRadius = Math.min(width, height) * 0.15;

    const animate = (time: number) => {
      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Spawn new particles
      if (time - lastSpawnRef.current > 50) {
        for (let i = 0; i < 3; i++) {
          particlesRef.current.push(spawnParticle(width, height));
        }
        lastSpawnRef.current = time;

        // Limit particles
        if (particlesRef.current.length > 150) {
          particlesRef.current = particlesRef.current.slice(-150);
        }
      }

      // Update and draw particles
      particlesRef.current = particlesRef.current.filter((p) => {
        // Update position
        p.x += p.vx;
        p.y += p.vy;
        p.life++;

        // Calculate distance to center
        const dx = centerX - p.x;
        const dy = centerY - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Fade out near core
        if (dist < coreRadius * 2) {
          p.opacity *= 0.95;
        }

        // Kill particle if at core or expired
        if (dist < coreRadius || p.life > p.maxLife || p.opacity < 0.05) {
          return false;
        }

        // Draw particle
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color.replace(")", `, ${p.opacity})`).replace("rgb", "rgba");
        ctx.fill();

        // Draw trail
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x - p.vx * 3, p.y - p.vy * 3);
        ctx.strokeStyle = p.color.replace(")", `, ${p.opacity * 0.5})`).replace("rgb", "rgba");
        ctx.lineWidth = p.size * 0.5;
        ctx.stroke();

        return true;
      });

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [enabled, canvasRef, spawnParticle]);
}

// ============================================================================
// Orb Component
// ============================================================================

function ConsensusSphere({
  votes,
  size,
  animated,
}: {
  votes: ModelVotes;
  size: number;
  animated: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [pulseScale, setPulseScale] = useState(1);

  useParticleSystem(canvasRef, votes, animated);

  // Pulse animation based on consensus strength
  useEffect(() => {
    if (!animated) return;

    const interval = setInterval(() => {
      setPulseScale((prev) => {
        const target = 1 + (votes.consensusStrength / 100) * 0.05;
        return prev === 1 ? target : 1;
      });
    }, 1500);

    return () => clearInterval(interval);
  }, [animated, votes.consensusStrength]);

  const consensusColor = getConsensusColor(votes.dominantSignal);
  const glowColor = getGlowColor(votes.dominantSignal, 0.6);

  // Calculate arc angles for the vote distribution ring
  const total = votes.totalModels;
  const bullishAngle = (votes.bullish / total) * Math.PI * 2;
  const bearishAngle = (votes.bearish / total) * Math.PI * 2;
  const neutralAngle = (votes.neutral / total) * Math.PI * 2;

  const createArc = (startAngle: number, endAngle: number, radius: number) => {
    const centerX = size / 2;
    const centerY = size / 2;
    const x1 = centerX + radius * Math.cos(startAngle - Math.PI / 2);
    const y1 = centerY + radius * Math.sin(startAngle - Math.PI / 2);
    const x2 = centerX + radius * Math.cos(endAngle - Math.PI / 2);
    const y2 = centerY + radius * Math.sin(endAngle - Math.PI / 2);
    const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;

    return `M ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2}`;
  };

  const ringRadius = size * 0.42;
  const coreRadius = size * 0.25;

  return (
    <div className="relative" style={{ width: size, height: size }}>
      {/* Particle Canvas */}
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        className="absolute inset-0 z-0"
      />

      {/* SVG Orb */}
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="absolute inset-0 z-10"
      >
        <defs>
          {/* Core gradient */}
          <radialGradient id="coreGradient" cx="40%" cy="40%" r="60%">
            <stop offset="0%" stopColor={consensusColor} stopOpacity="0.9" />
            <stop offset="50%" stopColor={consensusColor} stopOpacity="0.5" />
            <stop offset="100%" stopColor={consensusColor} stopOpacity="0.1" />
          </radialGradient>

          {/* Glow filter */}
          <filter id="orbGlow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="8" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          {/* Inner shadow */}
          <filter id="innerShadow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur in="SourceAlpha" stdDeviation="4" result="blur" />
            <feOffset in="blur" dx="2" dy="2" result="offsetBlur" />
            <feComposite
              in="SourceGraphic"
              in2="offsetBlur"
              operator="over"
              result="final"
            />
          </filter>
        </defs>

        {/* Background ring track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={ringRadius}
          fill="none"
          stroke="#262626"
          strokeWidth={12}
        />

        {/* Vote distribution ring - Bullish */}
        {votes.bullish > 0 && (
          <path
            d={createArc(0, bullishAngle, ringRadius)}
            fill="none"
            stroke="#22c55e"
            strokeWidth={12}
            strokeLinecap="round"
            className={cn(
              "transition-all duration-1000",
              animated && "animate-[fadeIn_0.5s_ease-out]"
            )}
          />
        )}

        {/* Vote distribution ring - Bearish */}
        {votes.bearish > 0 && (
          <path
            d={createArc(bullishAngle, bullishAngle + bearishAngle, ringRadius)}
            fill="none"
            stroke="#ef4444"
            strokeWidth={12}
            strokeLinecap="round"
            className={cn(
              "transition-all duration-1000",
              animated && "animate-[fadeIn_0.5s_ease-out_0.2s]"
            )}
          />
        )}

        {/* Vote distribution ring - Neutral */}
        {votes.neutral > 0 && (
          <path
            d={createArc(
              bullishAngle + bearishAngle,
              bullishAngle + bearishAngle + neutralAngle,
              ringRadius
            )}
            fill="none"
            stroke="#f59e0b"
            strokeWidth={12}
            strokeLinecap="round"
            className={cn(
              "transition-all duration-1000",
              animated && "animate-[fadeIn_0.5s_ease-out_0.4s]"
            )}
          />
        )}

        {/* Core orb with glow */}
        <g filter="url(#orbGlow)">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={coreRadius}
            fill="url(#coreGradient)"
            className="transition-all duration-300"
            style={{
              transform: `scale(${pulseScale})`,
              transformOrigin: "center",
            }}
          />
        </g>

        {/* Highlight on orb */}
        <ellipse
          cx={size / 2 - coreRadius * 0.3}
          cy={size / 2 - coreRadius * 0.3}
          rx={coreRadius * 0.25}
          ry={coreRadius * 0.15}
          fill="white"
          opacity="0.3"
        />

        {/* Consensus percentage in center */}
        <text
          x={size / 2}
          y={size / 2 - 8}
          textAnchor="middle"
          fill="white"
          fontSize={size * 0.1}
          fontWeight="bold"
          fontFamily="monospace"
        >
          {votes.consensusStrength.toFixed(0)}%
        </text>
        <text
          x={size / 2}
          y={size / 2 + 12}
          textAnchor="middle"
          fill="rgba(255,255,255,0.6)"
          fontSize={size * 0.04}
          fontFamily="sans-serif"
        >
          CONSENSUS
        </text>
      </svg>

      {/* Outer glow effect */}
      <div
        className="absolute inset-0 rounded-full opacity-30 blur-xl -z-10"
        style={{ backgroundColor: glowColor }}
      />
    </div>
  );
}

// ============================================================================
// Vote Detail Bar
// ============================================================================

function VoteDetailBar({
  label,
  count,
  total,
  color,
  icon: Icon,
}: {
  label: string;
  count: number;
  total: number;
  color: string;
  icon: React.ElementType;
}) {
  const percentage = (count / total) * 100;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4" style={{ color }} />
          <span className="text-xs font-medium text-neutral-300">{label}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-neutral-500">
            {count.toLocaleString()}
          </span>
          <span className="text-sm font-mono font-bold" style={{ color }}>
            {percentage.toFixed(1)}%
          </span>
        </div>
      </div>
      <div className="h-2 bg-neutral-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-1000 ease-out"
          style={{
            width: `${percentage}%`,
            backgroundColor: color,
            boxShadow: `0 0 10px ${color}40`,
          }}
        />
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function ModelConsensusOrb({
  votes,
  size = "md",
  showDetails = true,
  animated = true,
  onSignalClick,
}: ModelConsensusOrbProps) {
  const sizes = {
    sm: 180,
    md: 260,
    lg: 340,
  };

  const orbSize = sizes[size];
  const consensusColor = getConsensusColor(votes.dominantSignal);

  const SignalIcon =
    votes.dominantSignal === "bullish"
      ? TrendingUp
      : votes.dominantSignal === "bearish"
      ? TrendingDown
      : Minus;

  return (
    <div className="flex flex-col items-center gap-6">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div
          className="p-2 rounded-lg"
          style={{ backgroundColor: `${consensusColor}20` }}
        >
          <Users className="w-5 h-5" style={{ color: consensusColor }} />
        </div>
        <div>
          <div className="text-sm font-semibold text-neutral-200 flex items-center gap-2">
            Model Consensus
            <span
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold"
              style={{
                backgroundColor: `${consensusColor}20`,
                color: consensusColor,
              }}
            >
              <SignalIcon className="w-3 h-3" />
              {votes.dominantSignal.toUpperCase()}
            </span>
          </div>
          <div className="text-xs text-neutral-500 flex items-center gap-2">
            <Activity className="w-3 h-3 animate-pulse text-green-500" />
            {votes.totalModels.toLocaleString()} models voting
          </div>
        </div>
      </div>

      {/* Orb */}
      <div className="relative">
        <ConsensusSphere votes={votes} size={orbSize} animated={animated} />

        {/* Signal labels around orb */}
        <div
          className={cn(
            "absolute -top-1 left-1/2 -translate-x-1/2",
            "flex items-center gap-1.5 px-2 py-1 rounded-full",
            "bg-neutral-900/90 border border-green-500/30",
            "cursor-pointer hover:bg-green-500/10 transition-colors"
          )}
          onClick={() => onSignalClick?.("bullish")}
        >
          <TrendingUp className="w-3 h-3 text-green-400" />
          <span className="text-xs font-mono text-green-400">
            {((votes.bullish / votes.totalModels) * 100).toFixed(0)}%
          </span>
        </div>

        <div
          className={cn(
            "absolute top-1/2 -right-2 -translate-y-1/2",
            "flex items-center gap-1.5 px-2 py-1 rounded-full",
            "bg-neutral-900/90 border border-amber-500/30",
            "cursor-pointer hover:bg-amber-500/10 transition-colors"
          )}
          onClick={() => onSignalClick?.("neutral")}
        >
          <Minus className="w-3 h-3 text-amber-400" />
          <span className="text-xs font-mono text-amber-400">
            {((votes.neutral / votes.totalModels) * 100).toFixed(0)}%
          </span>
        </div>

        <div
          className={cn(
            "absolute -bottom-1 left-1/2 -translate-x-1/2",
            "flex items-center gap-1.5 px-2 py-1 rounded-full",
            "bg-neutral-900/90 border border-red-500/30",
            "cursor-pointer hover:bg-red-500/10 transition-colors"
          )}
          onClick={() => onSignalClick?.("bearish")}
        >
          <TrendingDown className="w-3 h-3 text-red-400" />
          <span className="text-xs font-mono text-red-400">
            {((votes.bearish / votes.totalModels) * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Details */}
      {showDetails && (
        <div className="w-full max-w-xs space-y-3 p-4 bg-neutral-900/50 rounded-xl border border-neutral-800">
          <VoteDetailBar
            label="Bullish"
            count={votes.bullish}
            total={votes.totalModels}
            color="#22c55e"
            icon={TrendingUp}
          />
          <VoteDetailBar
            label="Bearish"
            count={votes.bearish}
            total={votes.totalModels}
            color="#ef4444"
            icon={TrendingDown}
          />
          <VoteDetailBar
            label="Neutral"
            count={votes.neutral}
            total={votes.totalModels}
            color="#f59e0b"
            icon={Minus}
          />

          <div className="pt-2 border-t border-neutral-800">
            <div className="flex items-center justify-between text-xs">
              <span className="text-neutral-500">Consensus Strength</span>
              <div className="flex items-center gap-2">
                <Zap
                  className={cn(
                    "w-3.5 h-3.5",
                    votes.consensusStrength >= 70
                      ? "text-green-400"
                      : votes.consensusStrength >= 50
                      ? "text-amber-400"
                      : "text-red-400"
                  )}
                />
                <span
                  className={cn(
                    "font-mono font-bold",
                    votes.consensusStrength >= 70
                      ? "text-green-400"
                      : votes.consensusStrength >= 50
                      ? "text-amber-400"
                      : "text-red-400"
                  )}
                >
                  {votes.consensusStrength.toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Mock Data Generator
// ============================================================================

export function generateMockConsensus(
  direction: "bullish" | "bearish" | "neutral" = "bullish",
  strength: "strong" | "moderate" | "weak" = "strong"
): ModelVotes {
  const totalModels = 10179;
  const strengthMultiplier =
    strength === "strong" ? 0.7 : strength === "moderate" ? 0.5 : 0.35;

  let bullish: number, bearish: number, neutral: number;

  switch (direction) {
    case "bullish":
      bullish = Math.floor(totalModels * strengthMultiplier + Math.random() * totalModels * 0.1);
      bearish = Math.floor((totalModels - bullish) * (0.3 + Math.random() * 0.2));
      neutral = totalModels - bullish - bearish;
      break;
    case "bearish":
      bearish = Math.floor(totalModels * strengthMultiplier + Math.random() * totalModels * 0.1);
      bullish = Math.floor((totalModels - bearish) * (0.3 + Math.random() * 0.2));
      neutral = totalModels - bullish - bearish;
      break;
    case "neutral":
      neutral = Math.floor(totalModels * 0.4);
      bullish = Math.floor((totalModels - neutral) * 0.5);
      bearish = totalModels - bullish - neutral;
      break;
  }

  const maxVotes = Math.max(bullish, bearish, neutral);
  const consensusStrength = (maxVotes / totalModels) * 100;

  return {
    bullish,
    bearish,
    neutral,
    totalModels,
    consensusStrength,
    dominantSignal: direction,
    lastUpdated: new Date(),
  };
}
