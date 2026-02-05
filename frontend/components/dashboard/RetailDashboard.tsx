"use client";

import { useState, useMemo } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { MOCK_ASSETS, MOCK_SIGNALS, type Horizon, type SignalData } from "@/lib/mock-data";
import type { AssetId } from "@/types";
import {
  Sparkles,
  ChevronLeft,
  ChevronRight,
  CircleDot,
  ThumbsUp,
  ThumbsDown,
  HelpCircle,
  DollarSign,
  Clock,
  Users,
  Smile,
  Meh,
  Frown,
  ArrowRight,
  Shield,
  Eye,
  Zap,
  Gift,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

type FriendlySignal = "looks-good" | "heads-up" | "wait-and-see";

interface SimpleSignal {
  assetId: AssetId;
  name: string;
  emoji: string;
  price: number;
  priceFormatted: string;
  signal: FriendlySignal;
  confidence: number;
  headline: string;
  explanation: string;
  timeframe: string;
  agreementPercent: number;
}

// ============================================================================
// Friendly Language Helpers
// ============================================================================

const ASSET_EMOJIS: Record<string, string> = {
  "crude-oil": "🛢️",
  "natural-gas": "🔥",
  "gold": "🥇",
  "silver": "🥈",
  "copper": "🔶",
  "bitcoin": "₿",
  "ethereum": "💎",
  "sp500": "📈",
};

function getFriendlySignal(signal: SignalData): FriendlySignal {
  if (signal.confidence < 55 || signal.direction === "neutral") return "wait-and-see";
  if (signal.direction === "bullish") return "looks-good";
  return "heads-up";
}

function getHeadline(signal: FriendlySignal, name: string): string {
  switch (signal) {
    case "looks-good":
      return `${name} looks promising!`;
    case "heads-up":
      return `${name} might dip soon`;
    default:
      return `${name} is uncertain`;
  }
}

function getExplanation(signal: FriendlySignal, confidence: number, agreementPercent: number): string {
  if (signal === "wait-and-see") {
    return `Our AI isn't sure which way this will go. ${agreementPercent}% of models agree - we'd suggest waiting for a clearer picture.`;
  }

  const strength = confidence >= 75 ? "pretty confident" : "somewhat confident";

  if (signal === "looks-good") {
    return `Our AI is ${strength} this could go up. ${agreementPercent}% of our models agree it's a good opportunity.`;
  }

  return `Our AI is ${strength} this might drop. ${agreementPercent}% of our models see potential downside.`;
}

function getTimeframeText(horizon: Horizon): string {
  switch (horizon) {
    case "D+1": return "Tomorrow";
    case "D+5": return "This Week";
    case "D+10": return "Next 2 Weeks";
    default: return "Soon";
  }
}

function getSimpleSignals(): SimpleSignal[] {
  const signals: SimpleSignal[] = [];
  const horizons: Horizon[] = ["D+1", "D+5", "D+10"];

  Object.entries(MOCK_ASSETS).forEach(([assetId, asset]) => {
    let bestSignal: SignalData | null = null;

    for (const horizon of horizons) {
      const signal = MOCK_SIGNALS[assetId as AssetId]?.[horizon];
      if (signal) {
        if (!bestSignal || (signal.direction !== "neutral" && signal.confidence > bestSignal.confidence)) {
          bestSignal = signal;
        }
      }
    }

    if (bestSignal) {
      const friendlySignal = getFriendlySignal(bestSignal);
      const agreementPercent = Math.round((bestSignal.modelsAgreeing / bestSignal.modelsTotal) * 100);

      signals.push({
        assetId: assetId as AssetId,
        name: asset.name,
        emoji: ASSET_EMOJIS[assetId] || "📊",
        price: asset.currentPrice,
        priceFormatted: asset.currentPrice >= 1000
          ? `$${(asset.currentPrice / 1000).toFixed(1)}k`
          : `$${asset.currentPrice.toFixed(2)}`,
        signal: friendlySignal,
        confidence: bestSignal.confidence,
        headline: getHeadline(friendlySignal, asset.name),
        explanation: getExplanation(friendlySignal, bestSignal.confidence, agreementPercent),
        timeframe: getTimeframeText(bestSignal.horizon),
        agreementPercent,
      });
    }
  });

  return signals.sort((a, b) => b.confidence - a.confidence);
}

// ============================================================================
// Welcome Header - Mobile Friendly
// ============================================================================

function WelcomeHeader() {
  const hour = new Date().getHours();
  const greeting = hour < 12 ? "Good morning" : hour < 17 ? "Good afternoon" : "Good evening";

  return (
    <div className="px-1">
      <div className="flex items-center gap-3 mb-2">
        <div className="p-2.5 bg-gradient-to-br from-orange-500 to-amber-500 rounded-xl shadow-lg shadow-orange-500/20">
          <Sparkles className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-neutral-100">{greeting}!</h1>
          <p className="text-sm text-neutral-400">Here&apos;s what&apos;s happening today</p>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Quick Summary Cards - Swipeable Row
// ============================================================================

interface QuickSummaryProps {
  signals: SimpleSignal[];
}

function QuickSummary({ signals }: QuickSummaryProps) {
  const looksGood = signals.filter(s => s.signal === "looks-good").length;
  const headsUp = signals.filter(s => s.signal === "heads-up").length;
  const waitAndSee = signals.filter(s => s.signal === "wait-and-see").length;

  const summaryCards = [
    {
      icon: Smile,
      label: "Looks Good",
      count: looksGood,
      color: "from-green-500/20 to-emerald-500/20",
      border: "border-green-500/30",
      iconColor: "text-green-400",
      textColor: "text-green-400",
    },
    {
      icon: Frown,
      label: "Heads Up",
      count: headsUp,
      color: "from-red-500/20 to-rose-500/20",
      border: "border-red-500/30",
      iconColor: "text-red-400",
      textColor: "text-red-400",
    },
    {
      icon: Meh,
      label: "Wait & See",
      count: waitAndSee,
      color: "from-amber-500/20 to-yellow-500/20",
      border: "border-amber-500/30",
      iconColor: "text-amber-400",
      textColor: "text-amber-400",
    },
  ];

  return (
    <div className="overflow-x-auto pb-2 -mx-4 px-4 scrollbar-hide">
      <div className="flex gap-3 min-w-max">
        {summaryCards.map((card) => (
          <div
            key={card.label}
            className={cn(
              "flex items-center gap-3 px-4 py-3 rounded-xl border",
              `bg-gradient-to-r ${card.color}`,
              card.border
            )}
          >
            <card.icon className={cn("w-8 h-8", card.iconColor)} />
            <div>
              <div className={cn("text-2xl font-bold font-mono", card.textColor)}>
                {card.count}
              </div>
              <div className="text-xs text-neutral-400">{card.label}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Hero Signal Card - The Main Featured Signal
// ============================================================================

interface HeroSignalProps {
  signal: SimpleSignal;
}

function HeroSignalCard({ signal }: HeroSignalProps) {
  const signalConfig = {
    "looks-good": {
      bg: "from-green-500/10 via-emerald-500/5 to-green-500/10",
      border: "border-green-500/30",
      badge: "bg-green-500/20 text-green-400 border-green-500/30",
      badgeText: "Looks Good",
      icon: ThumbsUp,
      iconBg: "bg-green-500/20",
      iconColor: "text-green-400",
      barColor: "bg-green-500",
    },
    "heads-up": {
      bg: "from-red-500/10 via-rose-500/5 to-red-500/10",
      border: "border-red-500/30",
      badge: "bg-red-500/20 text-red-400 border-red-500/30",
      badgeText: "Heads Up",
      icon: ThumbsDown,
      iconBg: "bg-red-500/20",
      iconColor: "text-red-400",
      barColor: "bg-red-500",
    },
    "wait-and-see": {
      bg: "from-amber-500/10 via-yellow-500/5 to-amber-500/10",
      border: "border-amber-500/30",
      badge: "bg-amber-500/20 text-amber-400 border-amber-500/30",
      badgeText: "Wait & See",
      icon: HelpCircle,
      iconBg: "bg-amber-500/20",
      iconColor: "text-amber-400",
      barColor: "bg-amber-500",
    },
  };

  const config = signalConfig[signal.signal];
  const Icon = config.icon;

  return (
    <Card className={cn(
      "border-2 overflow-hidden",
      `bg-gradient-to-br ${config.bg}`,
      config.border
    )}>
      <CardContent className="p-5">
        {/* Top Badge */}
        <div className="flex items-center justify-between mb-4">
          <Badge className="bg-orange-500/20 text-orange-400 border-orange-500/30 px-3 py-1">
            <Gift className="w-3.5 h-3.5 mr-1.5" />
            Top Pick Today
          </Badge>
          <Badge className={cn("px-3 py-1 border", config.badge)}>
            {config.badgeText}
          </Badge>
        </div>

        {/* Asset Info */}
        <div className="flex items-center gap-4 mb-4">
          <div className="text-4xl">{signal.emoji}</div>
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-neutral-100">{signal.name}</h2>
            <div className="flex items-center gap-2 text-neutral-400">
              <DollarSign className="w-4 h-4" />
              <span className="text-lg">{signal.priceFormatted}</span>
            </div>
          </div>
          <div className={cn("p-3 rounded-xl", config.iconBg)}>
            <Icon className={cn("w-8 h-8", config.iconColor)} />
          </div>
        </div>

        {/* Headline */}
        <h3 className="text-xl font-semibold text-neutral-100 mb-2">
          {signal.headline}
        </h3>

        {/* Explanation */}
        <p className="text-neutral-300 mb-4 leading-relaxed">
          {signal.explanation}
        </p>

        {/* Confidence Bar */}
        <div className="mb-4">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-neutral-400">How sure are we?</span>
            <span className="font-mono font-bold text-neutral-200">{signal.confidence}%</span>
          </div>
          <div className="h-3 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className={cn("h-full rounded-full transition-all duration-500", config.barColor)}
              style={{ width: `${signal.confidence}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-neutral-500 mt-1">
            <span>Not Sure</span>
            <span>Very Sure</span>
          </div>
        </div>

        {/* Footer Info */}
        <div className="flex items-center justify-between pt-3 border-t border-neutral-700/50">
          <div className="flex items-center gap-2 text-sm text-neutral-400">
            <Clock className="w-4 h-4" />
            <span>Best for: {signal.timeframe}</span>
          </div>
          <div className="flex items-center gap-2 text-sm text-neutral-400">
            <Users className="w-4 h-4" />
            <span>{signal.agreementPercent}% agree</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Swipeable Signal Cards Carousel
// ============================================================================

interface SignalCarouselProps {
  signals: SimpleSignal[];
}

function SignalCarousel({ signals }: SignalCarouselProps) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const visibleSignals = signals.slice(1); // Skip first (hero)

  const goNext = () => setCurrentIndex((i) => Math.min(i + 1, visibleSignals.length - 1));
  const goPrev = () => setCurrentIndex((i) => Math.max(i - 1, 0));

  if (visibleSignals.length === 0) return null;

  return (
    <div className="space-y-4">
      {/* Section Header */}
      <div className="flex items-center justify-between px-1">
        <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
          <Eye className="w-5 h-5 text-orange-400" />
          More Opportunities
        </h2>
        <div className="flex items-center gap-2">
          <button
            onClick={goPrev}
            disabled={currentIndex === 0}
            className={cn(
              "p-2 rounded-lg border transition-all",
              currentIndex === 0
                ? "border-neutral-800 text-neutral-600"
                : "border-neutral-700 text-neutral-300 hover:bg-neutral-800 active:scale-95"
            )}
          >
            <ChevronLeft className="w-5 h-5" />
          </button>
          <button
            onClick={goNext}
            disabled={currentIndex === visibleSignals.length - 1}
            className={cn(
              "p-2 rounded-lg border transition-all",
              currentIndex === visibleSignals.length - 1
                ? "border-neutral-800 text-neutral-600"
                : "border-neutral-700 text-neutral-300 hover:bg-neutral-800 active:scale-95"
            )}
          >
            <ChevronRight className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Carousel */}
      <div className="overflow-hidden">
        <div
          className="flex transition-transform duration-300 ease-out gap-4"
          style={{ transform: `translateX(-${currentIndex * 100}%)` }}
        >
          {visibleSignals.map((signal) => (
            <div key={signal.assetId} className="flex-shrink-0 w-full">
              <SimpleSignalCard signal={signal} />
            </div>
          ))}
        </div>
      </div>

      {/* Dots Indicator */}
      <div className="flex justify-center gap-2">
        {visibleSignals.map((_, index) => (
          <button
            key={index}
            onClick={() => setCurrentIndex(index)}
            className={cn(
              "transition-all duration-200",
              index === currentIndex
                ? "w-6 h-2 bg-orange-500 rounded-full"
                : "w-2 h-2 bg-neutral-700 rounded-full hover:bg-neutral-600"
            )}
          />
        ))}
      </div>

      {/* Swipe Hint */}
      <p className="text-center text-xs text-neutral-500 flex items-center justify-center gap-1">
        <ChevronLeft className="w-3 h-3" />
        Swipe or tap arrows to see more
        <ChevronRight className="w-3 h-3" />
      </p>
    </div>
  );
}

// ============================================================================
// Simple Signal Card - Compact Version
// ============================================================================

interface SimpleSignalCardProps {
  signal: SimpleSignal;
}

function SimpleSignalCard({ signal }: SimpleSignalCardProps) {
  const signalConfig = {
    "looks-good": {
      bg: "bg-green-500/5",
      border: "border-green-500/20 hover:border-green-500/40",
      badge: "bg-green-500/20 text-green-400",
      badgeText: "Looks Good",
      icon: ThumbsUp,
      iconColor: "text-green-400",
      barColor: "bg-green-500",
    },
    "heads-up": {
      bg: "bg-red-500/5",
      border: "border-red-500/20 hover:border-red-500/40",
      badge: "bg-red-500/20 text-red-400",
      badgeText: "Heads Up",
      icon: ThumbsDown,
      iconColor: "text-red-400",
      barColor: "bg-red-500",
    },
    "wait-and-see": {
      bg: "bg-amber-500/5",
      border: "border-amber-500/20 hover:border-amber-500/40",
      badge: "bg-amber-500/20 text-amber-400",
      badgeText: "Wait & See",
      icon: HelpCircle,
      iconColor: "text-amber-400",
      barColor: "bg-amber-500",
    },
  };

  const config = signalConfig[signal.signal];
  const Icon = config.icon;

  return (
    <Card className={cn(
      "border transition-all cursor-pointer active:scale-[0.98]",
      config.bg,
      config.border
    )}>
      <CardContent className="p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <span className="text-2xl">{signal.emoji}</span>
            <div>
              <h3 className="font-semibold text-neutral-100">{signal.name}</h3>
              <span className="text-sm text-neutral-400">{signal.priceFormatted}</span>
            </div>
          </div>
          <div className={cn("p-2 rounded-lg", config.badge.split(" ")[0])}>
            <Icon className={cn("w-5 h-5", config.iconColor)} />
          </div>
        </div>

        {/* Status Badge */}
        <Badge className={cn("mb-3", config.badge)}>
          {config.badgeText}
        </Badge>

        {/* Headline */}
        <p className="text-sm text-neutral-300 mb-3">{signal.headline}</p>

        {/* Mini Confidence Bar */}
        <div className="flex items-center gap-3">
          <div className="flex-1 h-2 bg-neutral-800 rounded-full overflow-hidden">
            <div
              className={cn("h-full rounded-full", config.barColor)}
              style={{ width: `${signal.confidence}%` }}
            />
          </div>
          <span className="text-sm font-mono text-neutral-400">{signal.confidence}%</span>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between mt-3 pt-3 border-t border-neutral-800">
          <span className="text-xs text-neutral-500 flex items-center gap-1">
            <Clock className="w-3 h-3" />
            {signal.timeframe}
          </span>
          <span className="text-xs text-orange-400 flex items-center gap-1">
            See details
            <ArrowRight className="w-3 h-3" />
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Quick List View - All Signals at a Glance
// ============================================================================

interface QuickListProps {
  signals: SimpleSignal[];
}

function QuickListView({ signals }: QuickListProps) {
  const [showAll, setShowAll] = useState(false);
  const displaySignals = showAll ? signals : signals.slice(0, 5);

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between px-1">
        <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
          <Zap className="w-5 h-5 text-orange-400" />
          Quick View
        </h2>
        <span className="text-xs text-neutral-500">{signals.length} assets</span>
      </div>

      {/* List */}
      <div className="space-y-2">
        {displaySignals.map((signal) => (
          <QuickListItem key={signal.assetId} signal={signal} />
        ))}
      </div>

      {/* Show More Button */}
      {signals.length > 5 && (
        <button
          onClick={() => setShowAll(!showAll)}
          className="w-full py-3 text-sm text-orange-400 hover:text-orange-300
                     bg-neutral-900/50 border border-neutral-800 rounded-xl
                     hover:bg-neutral-800/50 transition-all active:scale-[0.99]"
        >
          {showAll ? "Show Less" : `Show All ${signals.length} Assets`}
        </button>
      )}
    </div>
  );
}

function QuickListItem({ signal }: { signal: SimpleSignal }) {
  const signalConfig = {
    "looks-good": {
      icon: ThumbsUp,
      color: "text-green-400",
      bg: "bg-green-500/10",
      label: "Good",
    },
    "heads-up": {
      icon: ThumbsDown,
      color: "text-red-400",
      bg: "bg-red-500/10",
      label: "Caution",
    },
    "wait-and-see": {
      icon: HelpCircle,
      color: "text-amber-400",
      bg: "bg-amber-500/10",
      label: "Wait",
    },
  };

  const config = signalConfig[signal.signal];
  const Icon = config.icon;

  return (
    <div className="flex items-center gap-3 p-3 bg-neutral-900/50 border border-neutral-800
                    rounded-xl hover:bg-neutral-800/50 transition-all cursor-pointer active:scale-[0.99]">
      {/* Emoji */}
      <span className="text-xl">{signal.emoji}</span>

      {/* Name & Price */}
      <div className="flex-1 min-w-0">
        <div className="font-medium text-neutral-100 truncate">{signal.name}</div>
        <div className="text-xs text-neutral-500">{signal.priceFormatted}</div>
      </div>

      {/* Confidence */}
      <div className="text-right mr-2">
        <div className="text-sm font-mono text-neutral-300">{signal.confidence}%</div>
        <div className="text-xs text-neutral-500">sure</div>
      </div>

      {/* Signal Icon */}
      <div className={cn("p-2 rounded-lg", config.bg)}>
        <Icon className={cn("w-4 h-4", config.color)} />
      </div>
    </div>
  );
}

// ============================================================================
// Beginner Tips Card
// ============================================================================

function BeginnerTips() {
  const tips = [
    { icon: ThumbsUp, text: "\"Looks Good\" = Our AI thinks it will go UP", color: "text-green-400" },
    { icon: ThumbsDown, text: "\"Heads Up\" = Our AI thinks it might go DOWN", color: "text-red-400" },
    { icon: HelpCircle, text: "\"Wait & See\" = No clear signal yet", color: "text-amber-400" },
  ];

  return (
    <Card className="bg-gradient-to-br from-orange-500/5 via-amber-500/5 to-orange-500/5 border-orange-500/20">
      <CardContent className="p-4">
        <h3 className="font-semibold text-neutral-100 mb-3 flex items-center gap-2">
          <HelpCircle className="w-5 h-5 text-orange-400" />
          What do these mean?
        </h3>
        <div className="space-y-2">
          {tips.map((tip, i) => (
            <div key={i} className="flex items-center gap-3 p-2 bg-neutral-900/30 rounded-lg">
              <tip.icon className={cn("w-5 h-5 flex-shrink-0", tip.color)} />
              <span className="text-sm text-neutral-300">{tip.text}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Safety Disclaimer - Friendly Version
// ============================================================================

function FriendlyDisclaimer() {
  return (
    <div className="p-4 bg-neutral-900/30 border border-neutral-800 rounded-xl">
      <div className="flex items-start gap-3">
        <Shield className="w-5 h-5 text-orange-400 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-neutral-400">
          <strong className="text-neutral-300">A friendly reminder:</strong> These are AI predictions to help you learn, not financial advice. Markets are unpredictable! Only invest money you&apos;re okay with losing, and chat with a financial advisor if you&apos;re unsure.
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Page Navigation Dots
// ============================================================================

interface PageDotsProps {
  current: number;
  total: number;
  onChange: (index: number) => void;
}

function PageDots({ current, total, onChange }: PageDotsProps) {
  return (
    <div className="flex justify-center gap-2 py-4">
      {Array.from({ length: total }, (_, i) => (
        <button
          key={i}
          onClick={() => onChange(i)}
          className={cn(
            "rounded-full transition-all",
            i === current
              ? "w-8 h-2 bg-orange-500"
              : "w-2 h-2 bg-neutral-700 hover:bg-neutral-600"
          )}
        />
      ))}
    </div>
  );
}

// ============================================================================
// Main Retail Dashboard - Mobile First Single Screen
// ============================================================================

export function RetailDashboard() {
  const signals = useMemo(() => getSimpleSignals(), []);
  const topSignal = signals[0];
  const [currentPage, setCurrentPage] = useState(0);

  const pages = [
    { id: "home", label: "Home" },
    { id: "list", label: "All" },
    { id: "tips", label: "Help" },
  ];

  return (
    <div className="max-w-lg mx-auto space-y-6 pb-8">
      {/* Welcome Header */}
      <WelcomeHeader />

      {/* Quick Summary - Horizontal Scroll */}
      <QuickSummary signals={signals} />

      {/* Page Navigation */}
      <PageDots current={currentPage} total={pages.length} onChange={setCurrentPage} />

      {/* Page Content */}
      {currentPage === 0 && (
        <div className="space-y-6">
          {/* Hero Signal Card */}
          {topSignal && <HeroSignalCard signal={topSignal} />}

          {/* More Signals Carousel */}
          <SignalCarousel signals={signals} />

          {/* Disclaimer */}
          <FriendlyDisclaimer />
        </div>
      )}

      {currentPage === 1 && (
        <div className="space-y-6">
          {/* Quick List View */}
          <QuickListView signals={signals} />

          {/* Disclaimer */}
          <FriendlyDisclaimer />
        </div>
      )}

      {currentPage === 2 && (
        <div className="space-y-6">
          {/* Beginner Tips */}
          <BeginnerTips />

          {/* More Help */}
          <Card className="bg-neutral-900/50 border-neutral-800">
            <CardContent className="p-4 space-y-4">
              <h3 className="font-semibold text-neutral-100">How to use this dashboard</h3>

              <div className="space-y-3 text-sm text-neutral-400">
                <div className="flex items-start gap-3">
                  <CircleDot className="w-4 h-4 text-orange-400 mt-0.5 flex-shrink-0" />
                  <span>The <strong className="text-neutral-200">Top Pick</strong> is what our AI is most confident about today</span>
                </div>
                <div className="flex items-start gap-3">
                  <CircleDot className="w-4 h-4 text-orange-400 mt-0.5 flex-shrink-0" />
                  <span>The <strong className="text-neutral-200">percentage</strong> shows how sure our AI is (higher = more confident)</span>
                </div>
                <div className="flex items-start gap-3">
                  <CircleDot className="w-4 h-4 text-orange-400 mt-0.5 flex-shrink-0" />
                  <span>Swipe left and right to see more opportunities</span>
                </div>
                <div className="flex items-start gap-3">
                  <CircleDot className="w-4 h-4 text-orange-400 mt-0.5 flex-shrink-0" />
                  <span>Tap the dots at the top to switch between views</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Disclaimer */}
          <FriendlyDisclaimer />
        </div>
      )}

      {/* Bottom Navigation Bar - Mobile Style */}
      <div className="fixed bottom-0 left-0 right-0 bg-neutral-950/95 backdrop-blur-lg border-t border-neutral-800 p-2 md:hidden">
        <div className="flex justify-around max-w-lg mx-auto">
          {pages.map((page, index) => (
            <button
              key={page.id}
              onClick={() => setCurrentPage(index)}
              className={cn(
                "flex flex-col items-center gap-1 px-6 py-2 rounded-lg transition-all",
                currentPage === index
                  ? "text-orange-400"
                  : "text-neutral-500 hover:text-neutral-300"
              )}
            >
              {index === 0 && <Sparkles className="w-5 h-5" />}
              {index === 1 && <Eye className="w-5 h-5" />}
              {index === 2 && <HelpCircle className="w-5 h-5" />}
              <span className="text-xs">{page.label}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
