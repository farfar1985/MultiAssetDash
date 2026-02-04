"use client";

import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { EnsembleMethod } from "@/lib/api-client";

interface EnsembleSelectorProps {
  value: EnsembleMethod;
  onChange: (method: EnsembleMethod) => void;
  variant?: "tabs" | "dropdown";
  className?: string;
}

interface MethodConfig {
  value: EnsembleMethod;
  label: string;
  shortLabel: string;
  description: string;
  badge?: string;
}

export const ENSEMBLE_METHODS: MethodConfig[] = [
  {
    value: "accuracy_weighted",
    label: "Accuracy Weighted",
    shortLabel: "Accuracy",
    description: "Weights models by historical directional accuracy",
  },
  {
    value: "exponential_decay",
    label: "Exponential Decay",
    shortLabel: "Exp Decay",
    description: "Recent predictions weighted more heavily",
  },
  {
    value: "top_k_sharpe",
    label: "Top-K Sharpe",
    shortLabel: "Top-K",
    description: "Uses only models with highest Sharpe ratios",
    badge: "BEST",
  },
  {
    value: "ridge_stacking",
    label: "Ridge Stacking",
    shortLabel: "Ridge",
    description: "L2-regularized meta-learning combination",
  },
  {
    value: "inverse_variance",
    label: "Inverse Variance",
    shortLabel: "Inv Var",
    description: "Lower variance models get higher weights",
  },
  {
    value: "pairwise_slope",
    label: "Pairwise Slope",
    shortLabel: "X-Horizon",
    description: "Cross-horizon trend analysis method",
  },
];

export function getMethodConfig(method: EnsembleMethod): MethodConfig {
  return ENSEMBLE_METHODS.find((m) => m.value === method) || ENSEMBLE_METHODS[0];
}

export function EnsembleSelector({
  value,
  onChange,
  variant = "tabs",
  className,
}: EnsembleSelectorProps) {
  if (variant === "dropdown") {
    const currentMethod = getMethodConfig(value);

    return (
      <Select value={value} onValueChange={(v) => onChange(v as EnsembleMethod)}>
        <SelectTrigger
          className={cn(
            "w-[200px] bg-neutral-900 border-neutral-700 text-sm",
            "hover:border-neutral-600 focus:ring-blue-500/20",
            className
          )}
        >
          <SelectValue>
            <span className="flex items-center gap-2">
              {currentMethod.shortLabel}
              {currentMethod.badge && (
                <Badge
                  variant="outline"
                  className="text-[10px] px-1.5 py-0 h-4 bg-green-500/10 border-green-500/30 text-green-500"
                >
                  {currentMethod.badge}
                </Badge>
              )}
            </span>
          </SelectValue>
        </SelectTrigger>
        <SelectContent className="bg-neutral-900 border-neutral-700">
          {ENSEMBLE_METHODS.map((method) => (
            <SelectItem
              key={method.value}
              value={method.value}
              className="text-sm focus:bg-neutral-800 focus:text-neutral-100"
            >
              <div className="flex flex-col gap-0.5">
                <div className="flex items-center gap-2">
                  <span>{method.label}</span>
                  {method.badge && (
                    <Badge
                      variant="outline"
                      className="text-[10px] px-1.5 py-0 h-4 bg-green-500/10 border-green-500/30 text-green-500"
                    >
                      {method.badge}
                    </Badge>
                  )}
                </div>
                <span className="text-xs text-neutral-500">
                  {method.description}
                </span>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  }

  return (
    <Tabs
      value={value}
      onValueChange={(v) => onChange(v as EnsembleMethod)}
      className={className}
    >
      <TabsList className="bg-neutral-900 border border-neutral-800 p-1 h-auto flex-wrap">
        {ENSEMBLE_METHODS.map((method) => (
          <TabsTrigger
            key={method.value}
            value={method.value}
            className={cn(
              "text-xs px-3 py-1.5 data-[state=active]:bg-neutral-800 data-[state=active]:text-neutral-100",
              "text-neutral-400 hover:text-neutral-200 transition-colors",
              "flex items-center gap-1.5"
            )}
            title={method.description}
          >
            {method.shortLabel}
            {method.badge && (
              <Badge
                variant="outline"
                className="text-[9px] px-1 py-0 h-3.5 bg-green-500/10 border-green-500/30 text-green-500"
              >
                {method.badge}
              </Badge>
            )}
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
