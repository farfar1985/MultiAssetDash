"use client";

import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { GitBranch, Info, TrendingUp, TrendingDown } from "lucide-react";

interface CorrelationData {
  assets: string[];
  matrix: number[][]; // Correlation values -1 to 1
}

interface CorrelationMatrixProps {
  data?: CorrelationData;
  size?: "sm" | "md" | "lg";
  showLabels?: boolean;
  interactive?: boolean;
  title?: string;
}

const DEFAULT_CORRELATION_DATA: CorrelationData = {
  assets: ["CL", "GC", "BTC", "SPY", "NG", "EUR"],
  matrix: [
    [1.0, 0.15, 0.32, 0.45, 0.62, -0.18],
    [0.15, 1.0, 0.28, -0.12, 0.08, 0.42],
    [0.32, 0.28, 1.0, 0.55, 0.22, -0.08],
    [0.45, -0.12, 0.55, 1.0, 0.18, -0.35],
    [0.62, 0.08, 0.22, 0.18, 1.0, -0.25],
    [-0.18, 0.42, -0.08, -0.35, -0.25, 1.0],
  ],
};

function getCorrelationColor(value: number): string {
  // Strong negative = red, neutral = gray, strong positive = green
  if (value >= 0.7) return "bg-green-500";
  if (value >= 0.4) return "bg-green-600/70";
  if (value >= 0.2) return "bg-green-700/50";
  if (value > -0.2) return "bg-neutral-700";
  if (value > -0.4) return "bg-red-700/50";
  if (value > -0.7) return "bg-red-600/70";
  return "bg-red-500";
}

function getCorrelationTextColor(value: number): string {
  if (Math.abs(value) >= 0.7) return "text-white";
  if (Math.abs(value) >= 0.4) return "text-neutral-200";
  return "text-neutral-400";
}

function CorrelationCell({
  value,
  rowAsset,
  colAsset,
  isDiagonal,
  size,
  isHighlighted,
  onHover,
  onClick,
}: {
  value: number;
  rowAsset: string;
  colAsset: string;
  isDiagonal: boolean;
  size: "sm" | "md" | "lg";
  isHighlighted: boolean;
  onHover?: (row: string, col: string) => void;
  onClick?: (row: string, col: string, value: number) => void;
}) {
  const cellSize = {
    sm: "w-10 h-10",
    md: "w-12 h-12",
    lg: "w-14 h-14",
  };

  const fontSize = {
    sm: "text-[10px]",
    md: "text-xs",
    lg: "text-sm",
  };

  if (isDiagonal) {
    return (
      <div
        className={cn(
          cellSize[size],
          "flex items-center justify-center",
          "bg-neutral-800 border border-neutral-700/50",
          "rounded-sm"
        )}
      >
        <span className="text-neutral-500 text-[10px]">1.00</span>
      </div>
    );
  }

  return (
    <div
      className={cn(
        cellSize[size],
        "flex items-center justify-center",
        "rounded-sm cursor-pointer transition-all duration-200",
        "border border-transparent",
        getCorrelationColor(value),
        isHighlighted && "ring-2 ring-white/50 scale-105 z-10",
        !isHighlighted && "hover:border-white/30 hover:scale-105"
      )}
      onMouseEnter={() => onHover?.(rowAsset, colAsset)}
      onMouseLeave={() => onHover?.("", "")}
      onClick={() => onClick?.(rowAsset, colAsset, value)}
      title={`${rowAsset}/${colAsset}: ${value.toFixed(2)}`}
    >
      <span className={cn("font-mono font-bold", fontSize[size], getCorrelationTextColor(value))}>
        {value.toFixed(2)}
      </span>
    </div>
  );
}

function CorrelationLegend() {
  return (
    <div className="flex items-center gap-2 mt-4 p-2 bg-neutral-800/30 rounded-lg">
      <span className="text-[10px] text-neutral-500 uppercase tracking-wider">Correlation:</span>
      <div className="flex items-center gap-1">
        <span className="text-[10px] text-red-400">Inverse</span>
        <div className="flex h-3 w-24 rounded overflow-hidden">
          <div className="flex-1 bg-red-500" />
          <div className="flex-1 bg-red-600/70" />
          <div className="flex-1 bg-neutral-700" />
          <div className="flex-1 bg-green-600/70" />
          <div className="flex-1 bg-green-500" />
        </div>
        <span className="text-[10px] text-green-400">Direct</span>
      </div>
    </div>
  );
}

export function CorrelationMatrix({
  data = DEFAULT_CORRELATION_DATA,
  size = "md",
  showLabels = true,
  interactive = true,
  title = "Asset Correlations",
}: CorrelationMatrixProps) {
  const [hoveredCell, setHoveredCell] = useState<{ row: string; col: string } | null>(null);
  const [selectedPair, setSelectedPair] = useState<{
    row: string;
    col: string;
    value: number;
  } | null>(null);

  const labelSize = {
    sm: "w-10 text-[10px]",
    md: "w-12 text-xs",
    lg: "w-14 text-sm",
  };

  // Calculate average correlations for insights
  const insights = useMemo(() => {
    const nonDiagonalValues: number[] = [];
    data.matrix.forEach((row, i) => {
      row.forEach((val, j) => {
        if (i !== j) nonDiagonalValues.push(val);
      });
    });

    const avgCorrelation =
      nonDiagonalValues.reduce((a, b) => a + b, 0) / nonDiagonalValues.length;

    // Find strongest correlations
    let maxCorr = { value: -2, pair: ["", ""] };
    let minCorr = { value: 2, pair: ["", ""] };

    data.matrix.forEach((row, i) => {
      row.forEach((val, j) => {
        if (i !== j) {
          if (val > maxCorr.value) {
            maxCorr = { value: val, pair: [data.assets[i], data.assets[j]] };
          }
          if (val < minCorr.value) {
            minCorr = { value: val, pair: [data.assets[i], data.assets[j]] };
          }
        }
      });
    });

    return { avgCorrelation, maxCorr, minCorr };
  }, [data]);

  const handleCellClick = (row: string, col: string, value: number) => {
    if (interactive) {
      setSelectedPair(selectedPair?.row === row && selectedPair?.col === col ? null : { row, col, value });
    }
  };

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="p-4 pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-500/10 rounded-lg border border-cyan-500/20">
              <GitBranch className="w-4 h-4 text-cyan-400" />
            </div>
            <div>
              <CardTitle className="text-sm font-semibold text-neutral-200">
                {title}
              </CardTitle>
              <span className="text-[10px] text-neutral-500">
                30-day rolling correlations
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Badge className="bg-blue-500/10 border-blue-500/30 text-blue-400 text-xs">
              {data.assets.length} Assets
            </Badge>
          </div>
        </div>
      </CardHeader>

      <CardContent className="p-4 pt-0">
        {/* Matrix Grid */}
        <div className="flex flex-col gap-0.5">
          {/* Header row with asset labels */}
          {showLabels && (
            <div className="flex gap-0.5">
              <div className={cn(labelSize[size], "h-8")} /> {/* Empty corner */}
              {data.assets.map((asset) => (
                <div
                  key={`header-${asset}`}
                  className={cn(
                    labelSize[size],
                    "h-8 flex items-center justify-center",
                    "font-mono font-bold text-neutral-400"
                  )}
                >
                  {asset}
                </div>
              ))}
            </div>
          )}

          {/* Matrix rows */}
          {data.matrix.map((row, rowIndex) => (
            <div key={`row-${rowIndex}`} className="flex gap-0.5">
              {/* Row label */}
              {showLabels && (
                <div
                  className={cn(
                    labelSize[size],
                    "flex items-center justify-center",
                    "font-mono font-bold text-neutral-400"
                  )}
                >
                  {data.assets[rowIndex]}
                </div>
              )}

              {/* Cells */}
              {row.map((value, colIndex) => {
                const isHighlighted =
                  (hoveredCell?.row === data.assets[rowIndex] &&
                    hoveredCell?.col === data.assets[colIndex]) ||
                  (selectedPair?.row === data.assets[rowIndex] &&
                    selectedPair?.col === data.assets[colIndex]);

                return (
                  <CorrelationCell
                    key={`cell-${rowIndex}-${colIndex}`}
                    value={value}
                    rowAsset={data.assets[rowIndex]}
                    colAsset={data.assets[colIndex]}
                    isDiagonal={rowIndex === colIndex}
                    size={size}
                    isHighlighted={isHighlighted}
                    onHover={
                      interactive
                        ? (r, c) => setHoveredCell(r && c ? { row: r, col: c } : null)
                        : undefined
                    }
                    onClick={interactive ? handleCellClick : undefined}
                  />
                );
              })}
            </div>
          ))}
        </div>

        {/* Legend */}
        <CorrelationLegend />

        {/* Quick Insights */}
        <div className="grid grid-cols-3 gap-2 mt-4">
          <div className="p-2 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <TrendingUp className="w-3 h-3 text-green-400" />
              <span className="text-[10px] uppercase tracking-wider text-neutral-500">
                Strongest +
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-green-400">
                {insights.maxCorr.pair.join("/")}
              </span>
              <span className="text-xs font-mono font-bold text-green-400">
                {insights.maxCorr.value.toFixed(2)}
              </span>
            </div>
          </div>

          <div className="p-2 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <TrendingDown className="w-3 h-3 text-red-400" />
              <span className="text-[10px] uppercase tracking-wider text-neutral-500">
                Strongest -
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-red-400">
                {insights.minCorr.pair.join("/")}
              </span>
              <span className="text-xs font-mono font-bold text-red-400">
                {insights.minCorr.value.toFixed(2)}
              </span>
            </div>
          </div>

          <div className="p-2 bg-neutral-800/30 rounded-lg border border-neutral-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <Info className="w-3 h-3 text-blue-400" />
              <span className="text-[10px] uppercase tracking-wider text-neutral-500">
                Average
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-neutral-400">Correlation</span>
              <span className="text-xs font-mono font-bold text-blue-400">
                {insights.avgCorrelation.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        {/* Selected pair detail */}
        {selectedPair && (
          <div className="mt-4 p-3 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 rounded-lg border border-cyan-500/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="font-mono font-bold text-cyan-400">
                  {selectedPair.row}/{selectedPair.col}
                </span>
                <span className="text-xs text-neutral-500">Correlation Pair</span>
              </div>
              <div
                className={cn(
                  "px-2 py-1 rounded font-mono font-bold text-sm",
                  selectedPair.value >= 0.4
                    ? "bg-green-500/20 text-green-400"
                    : selectedPair.value <= -0.4
                    ? "bg-red-500/20 text-red-400"
                    : "bg-neutral-700/50 text-neutral-300"
                )}
              >
                {selectedPair.value >= 0 ? "+" : ""}
                {selectedPair.value.toFixed(2)}
              </div>
            </div>
            <p className="text-xs text-neutral-400 mt-2">
              {selectedPair.value >= 0.7
                ? "Strong positive correlation - these assets tend to move together."
                : selectedPair.value >= 0.4
                ? "Moderate positive correlation - some tendency to move in same direction."
                : selectedPair.value <= -0.7
                ? "Strong negative correlation - these assets tend to move in opposite directions. Good for hedging."
                : selectedPair.value <= -0.4
                ? "Moderate negative correlation - some hedging potential."
                : "Weak correlation - movements are largely independent."}
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
