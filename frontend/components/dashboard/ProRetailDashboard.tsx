"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { MOCK_ASSETS, MOCK_SIGNALS, type Horizon, type SignalData } from "@/lib/mock-data";
import type { AssetId } from "@/types";
import {
  HMMRegimeIndicator,
  APIEnsembleConfidenceCard,
  type RegimeData,
} from "@/components/ensemble";
import { useEnsembleConfidence } from "@/hooks/useApi";
import { useHMMRegime } from "@/hooks";
import {
  GraduationCap,
  TrendingUp,
  TrendingDown,
  Minus,
  Lightbulb,
  BookOpen,
  Target,
  Shield,
  BarChart3,
  Brain,
  Layers,
  Info,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  Zap,
  Users,
  Clock,
  Scale,
  CircleDot,
  Activity,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface EnhancedSignal extends SignalData {
  name: string;
  symbol: string;
  currentPrice: number;
  change24h: number;
  changePercent24h: number;
  whySignal: WhySignalData;
}

interface WhySignalData {
  summary: string;
  keyFactors: SignalFactor[];
  riskFactors: string[];
  learnMore: LearnMoreItem[];
}

interface SignalFactor {
  name: string;
  impact: "positive" | "negative" | "neutral";
  description: string;
  contribution: number; // percentage contribution to signal
}

interface LearnMoreItem {
  term: string;
  definition: string;
  example: string;
}

// ============================================================================
// Educational Content Generators
// ============================================================================

function generateWhySignal(signal: SignalData, assetName: string): WhySignalData {
  const isHighConfidence = signal.confidence >= 75;
  const hasStrongAgreement = (signal.modelsAgreeing / signal.modelsTotal) > 0.75;

  // Generate contextual explanation
  const summaries = {
    bullish: isHighConfidence
      ? `Our AI models strongly believe ${assetName} will increase in value. ${signal.modelsAgreeing.toLocaleString()} out of ${signal.modelsTotal.toLocaleString()} models agree on this upward trend.`
      : `Our models lean toward ${assetName} going up, but confidence is moderate. Consider this a potential opportunity worth watching.`,
    bearish: isHighConfidence
      ? `Our AI models indicate ${assetName} may decline. This is a stronger signal suggesting caution or potential short opportunity.`
      : `Models suggest ${assetName} could drop, but the signal isn't overwhelming. Watch for confirmation before acting.`,
    neutral: `${assetName} shows no clear direction right now. Our models are split, meaning the market could go either way. Best to wait for a clearer signal.`,
  };

  // Key factors that drove this signal
  const keyFactors: SignalFactor[] = [];

  // Model Agreement Factor
  const agreementPercent = Math.round((signal.modelsAgreeing / signal.modelsTotal) * 100);
  keyFactors.push({
    name: "Model Consensus",
    impact: agreementPercent > 70 ? "positive" : agreementPercent > 50 ? "neutral" : "negative",
    description: `${agreementPercent}% of our AI models agree on this direction. Higher agreement means more confidence.`,
    contribution: 30,
  });

  // Sharpe Ratio Factor
  keyFactors.push({
    name: "Risk-Adjusted Return",
    impact: signal.sharpeRatio > 2 ? "positive" : signal.sharpeRatio > 1 ? "neutral" : "negative",
    description: signal.sharpeRatio > 2
      ? `Strong risk-adjusted returns (${signal.sharpeRatio.toFixed(2)}). The potential reward outweighs the risk.`
      : `Moderate risk-adjusted returns (${signal.sharpeRatio.toFixed(2)}). Returns are reasonable for the risk taken.`,
    contribution: 25,
  });

  // Directional Accuracy Factor
  keyFactors.push({
    name: "Historical Accuracy",
    impact: signal.directionalAccuracy > 58 ? "positive" : signal.directionalAccuracy > 52 ? "neutral" : "negative",
    description: `Our models have been right ${signal.directionalAccuracy.toFixed(1)}% of the time in similar conditions.`,
    contribution: 25,
  });

  // Confidence Factor
  keyFactors.push({
    name: "Signal Strength",
    impact: signal.confidence > 75 ? "positive" : signal.confidence > 60 ? "neutral" : "negative",
    description: signal.confidence > 75
      ? `High confidence signal (${signal.confidence}%). Models are quite certain.`
      : `Moderate confidence (${signal.confidence}%). Some uncertainty remains.`,
    contribution: 20,
  });

  // Risk factors
  const riskFactors: string[] = [];
  if (!isHighConfidence) {
    riskFactors.push("Confidence is below 75% - consider a smaller position size");
  }
  if (!hasStrongAgreement) {
    riskFactors.push("Model disagreement is notable - some AI models see it differently");
  }
  if (signal.horizon === "D+1") {
    riskFactors.push("Short timeframe (1 day) - requires quick decision making");
  }
  if (signal.horizon === "D+10") {
    riskFactors.push("Longer timeframe means more uncertainty - market conditions can change");
  }
  riskFactors.push("Past performance doesn't guarantee future results");
  riskFactors.push("Always invest only what you can afford to lose");

  // Educational learn more items
  const learnMore: LearnMoreItem[] = [
    {
      term: "Sharpe Ratio",
      definition: "Measures return per unit of risk. Higher is better. Above 2 is considered excellent.",
      example: `If Investment A returns 10% with 5% volatility (Sharpe = 2) and Investment B returns 10% with 10% volatility (Sharpe = 1), A is better because you get the same return with less risk.`,
    },
    {
      term: "Directional Accuracy",
      definition: "How often our models correctly predict if the price goes up or down.",
      example: `58% accuracy means if we make 100 predictions, about 58 are correct. In trading, even 55%+ consistently can be profitable.`,
    },
    {
      term: "Model Consensus",
      definition: "The percentage of our AI models that agree on the signal direction.",
      example: `If 8,000 out of 10,000 models say "bullish", that's 80% consensus - a strong agreement that increases confidence.`,
    },
    {
      term: "Time Horizon",
      definition: "How far into the future the prediction applies. D+1 = 1 day, D+5 = 5 days, D+10 = 10 days.",
      example: `A D+5 bullish signal means we expect the price to be higher in 5 days, not necessarily tomorrow.`,
    },
  ];

  return {
    summary: summaries[signal.direction],
    keyFactors,
    riskFactors: riskFactors.slice(0, 4),
    learnMore,
  };
}

function getEnhancedSignals(): EnhancedSignal[] {
  const signals: EnhancedSignal[] = [];
  const horizons: Horizon[] = ["D+1", "D+5", "D+10"];

  Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
    for (const horizon of horizons) {
      const signal = MOCK_SIGNALS[assetId as AssetId]?.[horizon];
      if (signal) {
        signals.push({
          ...signal,
          name: asset.name,
          symbol: asset.symbol,
          currentPrice: asset.currentPrice,
          change24h: asset.change24h,
          changePercent24h: asset.changePercent24h,
          whySignal: generateWhySignal(signal, asset.name),
        });
      }
    }
  });

  return signals.sort((a, b) => b.confidence - a.confidence);
}


// Default fallback regime data for loading/error states
const DEFAULT_REGIME: RegimeData = {
  regime: "sideways",
  confidence: 0.5,
  probabilities: { bull: 0.33, bear: 0.33, sideways: 0.34 },
  daysInRegime: 1,
  historicalAccuracy: 50,
  volatility: 20.0,
  trendStrength: 0.0,
};

// ============================================================================
// Components
// ============================================================================

function DashboardHeader() {
  return (
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-3">
        <div className="p-2.5 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl">
          <GraduationCap className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-neutral-100">Pro Retail Dashboard</h1>
          <p className="text-sm text-neutral-400">Learn while you trade - every signal explained</p>
        </div>
      </div>
      <Badge className="bg-cyan-500/20 text-cyan-400 border-cyan-500/30 px-3 py-1">
        <BookOpen className="w-3.5 h-3.5 mr-1.5" />
        Educational Mode
      </Badge>
    </div>
  );
}

// Quick Stats Bar
function QuickStats({ signals }: { signals: EnhancedSignal[] }) {
  const uniqueAssets = new Set(signals.map((s) => s.assetId)).size;
  const bullishCount = signals.filter((s) => s.direction === "bullish").length;
  const highConfidence = signals.filter((s) => s.confidence >= 75).length;
  const avgAccuracy = signals.reduce((sum, s) => sum + s.directionalAccuracy, 0) / signals.length;

  const stats = [
    { label: "Assets Tracked", value: uniqueAssets, icon: BarChart3, color: "text-cyan-400" },
    { label: "Bullish Signals", value: bullishCount, icon: TrendingUp, color: "text-green-400" },
    { label: "High Confidence", value: highConfidence, icon: Target, color: "text-purple-400" },
    { label: "Avg Accuracy", value: `${avgAccuracy.toFixed(1)}%`, icon: Activity, color: "text-amber-400" },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
      {stats.map((stat) => (
        <Card key={stat.label} className="bg-neutral-900/50 border-neutral-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-2 mb-1">
              <stat.icon className={cn("w-4 h-4", stat.color)} />
              <span className="text-xs text-neutral-500">{stat.label}</span>
            </div>
            <div className="text-2xl font-bold text-neutral-100">{stat.value}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// Educational Signal Card with "Why This Signal" Expansion
function EducationalSignalCard({ signal }: { signal: EnhancedSignal }) {
  const [expanded, setExpanded] = useState(false);
  const [showLearnMore, setShowLearnMore] = useState(false);

  const directionConfig = {
    bullish: {
      icon: TrendingUp,
      color: "text-green-400",
      bg: "bg-green-500/10",
      border: "border-green-500/20",
      badge: "bg-green-500/20 text-green-400",
      label: "Bullish",
    },
    bearish: {
      icon: TrendingDown,
      color: "text-red-400",
      bg: "bg-red-500/10",
      border: "border-red-500/20",
      badge: "bg-red-500/20 text-red-400",
      label: "Bearish",
    },
    neutral: {
      icon: Minus,
      color: "text-amber-400",
      bg: "bg-amber-500/10",
      border: "border-amber-500/20",
      badge: "bg-amber-500/20 text-amber-400",
      label: "Neutral",
    },
  };

  const config = directionConfig[signal.direction];
  const DirectionIcon = config.icon;
  const agreementPercent = Math.round((signal.modelsAgreeing / signal.modelsTotal) * 100);

  return (
    <Card className={cn("border overflow-hidden transition-all", config.bg, config.border, expanded && "ring-1 ring-cyan-500/30")}>
      <CardContent className="p-0">
        {/* Main Card Header */}
        <div className="p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className={cn("p-2 rounded-lg", config.bg)}>
                <DirectionIcon className={cn("w-5 h-5", config.color)} />
              </div>
              <div>
                <h3 className="font-semibold text-neutral-100">{signal.name}</h3>
                <div className="flex items-center gap-2 text-sm text-neutral-400">
                  <span>{signal.symbol}</span>
                  <span>•</span>
                  <span>${signal.currentPrice.toLocaleString()}</span>
                </div>
              </div>
            </div>
            <div className="text-right">
              <Badge className={cn("mb-1", config.badge)}>{config.label}</Badge>
              <div className="text-xs text-neutral-500">{signal.horizon}</div>
            </div>
          </div>

          {/* Key Metrics Row */}
          <div className="grid grid-cols-3 gap-3 mb-3">
            <div className="text-center p-2 bg-neutral-800/50 rounded-lg">
              <div className="text-lg font-bold text-neutral-100">{signal.confidence}%</div>
              <div className="text-xs text-neutral-500">Confidence</div>
            </div>
            <div className="text-center p-2 bg-neutral-800/50 rounded-lg">
              <div className="text-lg font-bold text-neutral-100">{agreementPercent}%</div>
              <div className="text-xs text-neutral-500">Agreement</div>
            </div>
            <div className="text-center p-2 bg-neutral-800/50 rounded-lg">
              <div className="text-lg font-bold text-neutral-100">{signal.sharpeRatio.toFixed(2)}</div>
              <div className="text-xs text-neutral-500">Sharpe</div>
            </div>
          </div>

          {/* Why This Signal Button */}
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full flex items-center justify-between p-3 bg-gradient-to-r from-cyan-500/10 to-blue-500/10
                       border border-cyan-500/20 rounded-lg hover:bg-cyan-500/15 transition-all group"
          >
            <div className="flex items-center gap-2">
              <Lightbulb className="w-4 h-4 text-cyan-400" />
              <span className="text-sm font-medium text-cyan-400">Why this signal?</span>
            </div>
            {expanded ? (
              <ChevronUp className="w-4 h-4 text-cyan-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-cyan-400 group-hover:translate-y-0.5 transition-transform" />
            )}
          </button>
        </div>

        {/* Expanded Educational Content */}
        {expanded && (
          <div className="border-t border-neutral-800 bg-neutral-900/50">
            {/* Summary */}
            <div className="p-4 border-b border-neutral-800">
              <div className="flex items-start gap-3">
                <div className="p-2 bg-cyan-500/10 rounded-lg shrink-0">
                  <Brain className="w-4 h-4 text-cyan-400" />
                </div>
                <div>
                  <h4 className="text-sm font-semibold text-neutral-200 mb-1">Signal Explanation</h4>
                  <p className="text-sm text-neutral-400 leading-relaxed">{signal.whySignal.summary}</p>
                </div>
              </div>
            </div>

            {/* Key Factors */}
            <div className="p-4 border-b border-neutral-800">
              <h4 className="text-sm font-semibold text-neutral-200 mb-3 flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                What&apos;s Driving This Signal
              </h4>
              <div className="space-y-3">
                {signal.whySignal.keyFactors.map((factor, idx) => (
                  <div key={idx} className="flex items-start gap-3">
                    <div className={cn(
                      "p-1.5 rounded-lg shrink-0",
                      factor.impact === "positive" ? "bg-green-500/10" :
                      factor.impact === "negative" ? "bg-red-500/10" : "bg-amber-500/10"
                    )}>
                      {factor.impact === "positive" ? (
                        <CheckCircle2 className="w-3.5 h-3.5 text-green-400" />
                      ) : factor.impact === "negative" ? (
                        <XCircle className="w-3.5 h-3.5 text-red-400" />
                      ) : (
                        <CircleDot className="w-3.5 h-3.5 text-amber-400" />
                      )}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-0.5">
                        <span className="text-sm font-medium text-neutral-200">{factor.name}</span>
                        <span className="text-xs text-neutral-500">{factor.contribution}% weight</span>
                      </div>
                      <p className="text-xs text-neutral-400">{factor.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Risk Factors */}
            <div className="p-4 border-b border-neutral-800">
              <h4 className="text-sm font-semibold text-neutral-200 mb-3 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-400" />
                Things to Consider
              </h4>
              <div className="space-y-2">
                {signal.whySignal.riskFactors.map((risk, idx) => (
                  <div key={idx} className="flex items-start gap-2 text-sm text-neutral-400">
                    <Shield className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                    <span>{risk}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Learn More Toggle */}
            <div className="p-4">
              <button
                onClick={() => setShowLearnMore(!showLearnMore)}
                className="flex items-center gap-2 text-sm text-cyan-400 hover:text-cyan-300 transition-colors"
              >
                <GraduationCap className="w-4 h-4" />
                <span>{showLearnMore ? "Hide" : "Show"} Glossary</span>
                {showLearnMore ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              </button>

              {showLearnMore && (
                <div className="mt-4 space-y-4">
                  {signal.whySignal.learnMore.map((item, idx) => (
                    <div key={idx} className="p-3 bg-neutral-800/50 rounded-lg">
                      <h5 className="text-sm font-semibold text-cyan-400 mb-1">{item.term}</h5>
                      <p className="text-xs text-neutral-300 mb-2">{item.definition}</p>
                      <div className="flex items-start gap-2 text-xs text-neutral-500">
                        <Info className="w-3 h-3 shrink-0 mt-0.5" />
                        <span>{item.example}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Learning Progress Sidebar
function LearningProgress() {
  const concepts = [
    { name: "Sharpe Ratio", learned: true },
    { name: "Model Consensus", learned: true },
    { name: "Directional Accuracy", learned: true },
    { name: "Time Horizons", learned: false },
    { name: "Position Sizing", learned: false },
    { name: "Risk Management", learned: false },
  ];

  const learnedCount = concepts.filter((c) => c.learned).length;

  return (
    <Card className="bg-gradient-to-br from-cyan-500/5 to-blue-500/5 border-cyan-500/20">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <GraduationCap className="w-5 h-5 text-cyan-400" />
          Learning Progress
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="mb-4">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-neutral-400">Concepts Explored</span>
            <span className="text-cyan-400 font-medium">{learnedCount}/{concepts.length}</span>
          </div>
          <div className="h-2 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all"
              style={{ width: `${(learnedCount / concepts.length) * 100}%` }}
            />
          </div>
        </div>

        <div className="space-y-2">
          {concepts.map((concept) => (
            <div
              key={concept.name}
              className={cn(
                "flex items-center gap-2 p-2 rounded-lg text-sm",
                concept.learned ? "bg-cyan-500/10" : "bg-neutral-800/50"
              )}
            >
              {concept.learned ? (
                <CheckCircle2 className="w-4 h-4 text-cyan-400" />
              ) : (
                <CircleDot className="w-4 h-4 text-neutral-500" />
              )}
              <span className={concept.learned ? "text-neutral-200" : "text-neutral-500"}>
                {concept.name}
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// Quick Tips Card
function QuickTips() {
  const tips = [
    {
      icon: Target,
      tip: "High confidence (>75%) signals are generally more reliable",
      color: "text-green-400",
    },
    {
      icon: Users,
      tip: "Look for high model agreement - more models = more perspectives",
      color: "text-blue-400",
    },
    {
      icon: Scale,
      tip: "Never invest more than you can afford to lose",
      color: "text-amber-400",
    },
    {
      icon: Clock,
      tip: "Match your trading style to the time horizon",
      color: "text-purple-400",
    },
  ];

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Lightbulb className="w-5 h-5 text-amber-400" />
          Pro Tips
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0 space-y-3">
        {tips.map((item, idx) => (
          <div key={idx} className="flex items-start gap-3">
            <item.icon className={cn("w-4 h-4 shrink-0 mt-0.5", item.color)} />
            <span className="text-sm text-neutral-400">{item.tip}</span>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

// Filter Controls
function FilterControls({
  direction,
  setDirection,
  horizon,
  setHorizon,
  minConfidence,
  setMinConfidence,
}: {
  direction: string;
  setDirection: (d: string) => void;
  horizon: string;
  setHorizon: (h: string) => void;
  minConfidence: number;
  setMinConfidence: (c: number) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3 mb-4">
      {/* Direction Filter */}
      <div className="flex items-center gap-1 bg-neutral-900 rounded-lg p-1">
        {["all", "bullish", "bearish"].map((d) => (
          <button
            key={d}
            onClick={() => setDirection(d)}
            className={cn(
              "px-3 py-1.5 text-xs font-medium rounded-md transition-all capitalize",
              direction === d
                ? "bg-neutral-700 text-neutral-100"
                : "text-neutral-400 hover:text-neutral-200"
            )}
          >
            {d === "all" ? "All" : d}
          </button>
        ))}
      </div>

      {/* Horizon Filter */}
      <div className="flex items-center gap-1 bg-neutral-900 rounded-lg p-1">
        {["all", "D+1", "D+5", "D+10"].map((h) => (
          <button
            key={h}
            onClick={() => setHorizon(h)}
            className={cn(
              "px-3 py-1.5 text-xs font-medium rounded-md transition-all",
              horizon === h
                ? "bg-neutral-700 text-neutral-100"
                : "text-neutral-400 hover:text-neutral-200"
            )}
          >
            {h === "all" ? "All" : h}
          </button>
        ))}
      </div>

      {/* Confidence Filter */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-neutral-500">Min Confidence:</span>
        <select
          value={minConfidence}
          onChange={(e) => setMinConfidence(Number(e.target.value))}
          className="bg-neutral-900 border border-neutral-700 rounded-lg px-2 py-1.5 text-xs text-neutral-200"
        >
          <option value={0}>Any</option>
          <option value={60}>60%+</option>
          <option value={70}>70%+</option>
          <option value={80}>80%+</option>
        </select>
      </div>
    </div>
  );
}

// ============================================================================
// Main Dashboard
// ============================================================================

export function ProRetailDashboard() {
  const allSignals = useMemo(() => getEnhancedSignals(), []);
  const [direction, setDirection] = useState("all");
  const [horizon, setHorizon] = useState("all");
  const [minConfidence, setMinConfidence] = useState(0);

  // Fetch HMM regime data from API
  const { data: regimeData = DEFAULT_REGIME } = useHMMRegime("crude-oil");
  // Fetch ensemble confidence for display explanation
  const { data: ensembleConfidence } = useEnsembleConfidence("crude-oil");

  const filteredSignals = useMemo(() => {
    return allSignals.filter((signal) => {
      if (direction !== "all" && signal.direction !== direction) return false;
      if (horizon !== "all" && signal.horizon !== horizon) return false;
      if (signal.confidence < minConfidence) return false;
      return true;
    });
  }, [allSignals, direction, horizon, minConfidence]);

  // Get unique top signals (one per asset, highest confidence)
  const topSignals = useMemo(() => {
    const byAsset = new Map<AssetId, EnhancedSignal>();
    for (const signal of filteredSignals) {
      const existing = byAsset.get(signal.assetId);
      if (!existing || signal.confidence > existing.confidence) {
        byAsset.set(signal.assetId, signal);
      }
    }
    return Array.from(byAsset.values())
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 6);
  }, [filteredSignals]);

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <DashboardHeader />

      {/* Quick Stats */}
      <QuickStats signals={allSignals} />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Signal Cards - Main Content */}
        <div className="lg:col-span-3 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
              <Zap className="w-5 h-5 text-cyan-400" />
              Educational Signals
              <Badge variant="outline" className="text-neutral-400 border-neutral-700">
                {filteredSignals.length} signals
              </Badge>
            </h2>
          </div>

          <FilterControls
            direction={direction}
            setDirection={setDirection}
            horizon={horizon}
            setHorizon={setHorizon}
            minConfidence={minConfidence}
            setMinConfidence={setMinConfidence}
          />

          {/* Signal Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {topSignals.map((signal) => (
              <EducationalSignalCard key={`${signal.assetId}-${signal.horizon}`} signal={signal} />
            ))}
          </div>

          {topSignals.length === 0 && (
            <Card className="bg-neutral-900/50 border-neutral-800">
              <CardContent className="p-8 text-center">
                <HelpCircle className="w-12 h-12 text-neutral-600 mx-auto mb-3" />
                <p className="text-neutral-400">No signals match your filters.</p>
                <p className="text-sm text-neutral-500 mt-1">Try adjusting your filter criteria.</p>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Simplified Ensemble Confidence - Educational - API Connected */}
          <Card className="bg-gradient-to-br from-purple-500/5 to-cyan-500/5 border-purple-500/20">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                AI Confidence
                <Badge className="bg-purple-500/20 text-purple-400 border-purple-500/30 text-[9px] ml-auto">
                  Live
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <APIEnsembleConfidenceCard
                assetId="crude-oil"
                showBreakdown={false}
                compact={true}
              />
              <div className="mt-3 p-2 bg-neutral-800/30 rounded-lg">
                <p className="text-xs text-neutral-400">
                  <strong className="text-purple-400">What this means:</strong> Our AI models are{" "}
                  {ensembleConfidence?.confidence ?? "--"}% confident in a{" "}
                  <span className={ensembleConfidence?.direction === "bullish" ? "text-green-400" : ensembleConfidence?.direction === "bearish" ? "text-red-400" : "text-amber-400"}>
                    {ensembleConfidence?.direction ?? "calculating"}
                  </span>{" "}
                  market direction.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Simplified Market Regime */}
          <Card className="bg-gradient-to-br from-cyan-500/5 to-blue-500/5 border-cyan-500/20">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Activity className="w-5 h-5 text-cyan-400" />
                Market Mood
              </CardTitle>
            </CardHeader>
            <CardContent className="pt-0">
              <HMMRegimeIndicator
                assetId="crude-oil"
                showProbabilities={false}
                compact={true}
                size="sm"
              />
              <div className="mt-3 p-2 bg-neutral-800/30 rounded-lg">
                <p className="text-xs text-neutral-400">
                  <strong className="text-cyan-400">Current market:</strong> We&apos;re in a{" "}
                  <span className={
                    regimeData.regime === "bull" ? "text-green-400" :
                    regimeData.regime === "bear" ? "text-red-400" : "text-amber-400"
                  }>
                    {regimeData.regime === "bull" ? "bullish" :
                     regimeData.regime === "bear" ? "bearish" : "sideways"}
                  </span>{" "}
                  market for the past {regimeData.daysInRegime} days.
                </p>
              </div>
            </CardContent>
          </Card>

          <LearningProgress />
          <QuickTips />

          {/* Disclaimer */}
          <Card className="bg-amber-500/5 border-amber-500/20">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <Shield className="w-5 h-5 text-amber-400 shrink-0" />
                <div className="text-xs text-neutral-400">
                  <strong className="text-amber-400">Educational Purpose:</strong> These signals are for learning. Always do your own research and consider consulting a financial advisor.
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
