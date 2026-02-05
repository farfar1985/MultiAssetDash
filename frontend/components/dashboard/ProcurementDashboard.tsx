"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { MOCK_ASSETS, MOCK_SIGNALS, type SignalData } from "@/lib/mock-data";
import type { AssetId } from "@/types";
import {
  Truck,
  Package,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  Clock,
  Calendar,
  CheckCircle2,
  Target,
  Factory,
  Globe,
  Warehouse,
  Scale,
  Zap,
  Bell,
  Shield,
  Activity,
  CircleDot,
  Layers,
  FileText,
  Calculator,
  PiggyBank,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface CommodityData {
  assetId: AssetId;
  name: string;
  symbol: string;
  currentPrice: number;
  unit: string;
  change24h: number;
  changePercent24h: number;
  signal: SignalData;
  procurement: ProcurementMetrics;
}

interface ProcurementMetrics {
  buyWindow: "optimal" | "acceptable" | "wait" | "urgent";
  priceOutlook: "falling" | "stable" | "rising";
  supplyRisk: "low" | "medium" | "high";
  recommendedAction: string;
  savingsOpportunity: number; // percentage
  contractTiming: string;
  volatilityLevel: "low" | "medium" | "high";
  hedgeRecommendation: string;
}

interface BuyingOpportunity {
  assetId: AssetId;
  name: string;
  symbol: string;
  currentPrice: number;
  unit: string;
  urgency: "buy-now" | "consider" | "monitor" | "wait";
  expectedMove: number;
  timeframe: string;
  savingsPotential: number;
  confidence: number;
  reasoning: string;
}

// ============================================================================
// Data Generators
// ============================================================================

const COMMODITY_UNITS: Record<string, string> = {
  "crude-oil": "/barrel",
  "natural-gas": "/MMBtu",
  gold: "/oz",
  silver: "/oz",
  copper: "/lb",
  platinum: "/oz",
  wheat: "/bushel",
  corn: "/bushel",
  soybean: "/bushel",
  bitcoin: "",
};

function getProcurementMetrics(signal: SignalData): ProcurementMetrics {
  const isBullish = signal.direction === "bullish";
  const isBearish = signal.direction === "bearish";
  const highConfidence = signal.confidence >= 70;

  // Determine buy window
  let buyWindow: ProcurementMetrics["buyWindow"] = "acceptable";
  if (isBearish && highConfidence) {
    buyWindow = "optimal"; // Prices expected to fall - wait
  } else if (isBullish && highConfidence) {
    buyWindow = "urgent"; // Prices expected to rise - buy now
  } else if (signal.direction === "neutral") {
    buyWindow = "acceptable";
  } else {
    buyWindow = isBullish ? "acceptable" : "wait";
  }

  // Price outlook
  const priceOutlook: ProcurementMetrics["priceOutlook"] = isBullish
    ? "rising"
    : isBearish
    ? "falling"
    : "stable";

  // Supply risk (derived from volatility/confidence)
  const supplyRisk: ProcurementMetrics["supplyRisk"] =
    signal.confidence < 55 ? "high" : signal.confidence < 70 ? "medium" : "low";

  // Recommended action
  const actions = {
    optimal: "Hold off purchasing - prices may decline further",
    urgent: "Lock in prices now before expected increase",
    consider: "Good time to fulfill near-term requirements",
    acceptable: "Standard purchasing conditions apply",
    wait: "Monitor closely, better opportunities may arise",
  };

  // Savings opportunity
  const savingsOpportunity = isBearish
    ? Math.round(5 + Math.random() * 10)
    : isBullish
    ? -Math.round(3 + Math.random() * 8)
    : Math.round(-2 + Math.random() * 4);

  // Contract timing
  const contractTiming = isBullish
    ? "Consider longer-term contracts to lock current rates"
    : isBearish
    ? "Short-term contracts recommended - better prices ahead"
    : "Standard contract terms appropriate";

  // Volatility
  const volatilityLevel: ProcurementMetrics["volatilityLevel"] =
    signal.sharpeRatio > 2.5 ? "high" : signal.sharpeRatio > 1.5 ? "medium" : "low";

  // Hedge recommendation
  const hedgeRecommendation = isBullish
    ? "Consider hedging against price increases"
    : isBearish
    ? "Hedging may not be necessary short-term"
    : "Monitor market for hedging opportunities";

  return {
    buyWindow,
    priceOutlook,
    supplyRisk,
    recommendedAction: actions[buyWindow],
    savingsOpportunity,
    contractTiming,
    volatilityLevel,
    hedgeRecommendation,
  };
}

function getCommodityData(): CommodityData[] {
  const commodities: CommodityData[] = [];

  // Filter out bitcoin for procurement (not a physical commodity)
  const procurementAssets = Object.entries(MOCK_ASSETS).filter(
    ([id]) => id !== "bitcoin"
  );

  procurementAssets.forEach(([assetId, asset]) => {
    const signal = MOCK_SIGNALS[assetId as AssetId]?.["D+5"]; // Use 5-day outlook for procurement
    if (signal) {
      commodities.push({
        assetId: assetId as AssetId,
        name: asset.name,
        symbol: asset.symbol,
        currentPrice: asset.currentPrice,
        unit: COMMODITY_UNITS[assetId] || "",
        change24h: asset.change24h,
        changePercent24h: asset.changePercent24h,
        signal,
        procurement: getProcurementMetrics(signal),
      });
    }
  });

  // Sort by buy window urgency
  const urgencyOrder = { urgent: 0, optimal: 1, consider: 2, acceptable: 3, wait: 4 };
  return commodities.sort(
    (a, b) => urgencyOrder[a.procurement.buyWindow] - urgencyOrder[b.procurement.buyWindow]
  );
}

function getBuyingOpportunities(commodities: CommodityData[]): BuyingOpportunity[] {
  return commodities
    .filter((c) => c.procurement.buyWindow === "optimal" || c.procurement.buyWindow === "urgent")
    .map((c) => {
      const urgency: BuyingOpportunity["urgency"] =
        c.procurement.buyWindow === "urgent"
          ? "buy-now"
          : c.procurement.buyWindow === "optimal"
          ? "consider"
          : "monitor";

      return {
        assetId: c.assetId,
        name: c.name,
        symbol: c.symbol,
        currentPrice: c.currentPrice,
        unit: c.unit,
        urgency,
        expectedMove: c.signal.direction === "bullish" ? 5 + Math.random() * 8 : -(3 + Math.random() * 6),
        timeframe: c.signal.horizon === "D+1" ? "1 day" : c.signal.horizon === "D+5" ? "5 days" : "10 days",
        savingsPotential: Math.abs(c.procurement.savingsOpportunity),
        confidence: c.signal.confidence,
        reasoning:
          c.signal.direction === "bullish"
            ? `Prices expected to rise ${(5 + Math.random() * 5).toFixed(1)}% - secure supply now`
            : `Potential ${(3 + Math.random() * 5).toFixed(1)}% savings if you wait for price dip`,
      };
    })
    .slice(0, 4);
}

// ============================================================================
// Components
// ============================================================================

function DashboardHeader() {
  return (
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-3">
        <div className="p-2.5 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl">
          <Truck className="w-6 h-6 text-white" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-neutral-100">Procurement Command Center</h1>
          <p className="text-sm text-neutral-400">Strategic sourcing & supply chain intelligence</p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30 px-3 py-1">
          <Activity className="w-3.5 h-3.5 mr-1.5" />
          Live Pricing
        </Badge>
      </div>
    </div>
  );
}

// Market Overview Stats
function MarketOverview({ commodities }: { commodities: CommodityData[] }) {
  const urgentBuys = commodities.filter((c) => c.procurement.buyWindow === "urgent").length;
  const optimalWindows = commodities.filter((c) => c.procurement.buyWindow === "optimal").length;
  const highRisk = commodities.filter((c) => c.procurement.supplyRisk === "high").length;
  const avgSavings =
    commodities.reduce((sum, c) => sum + Math.max(0, c.procurement.savingsOpportunity), 0) /
    commodities.length;

  const stats = [
    {
      label: "Action Required",
      value: urgentBuys,
      subtext: "commodities",
      icon: Bell,
      color: "text-red-400",
      bgColor: "bg-red-500/10",
    },
    {
      label: "Buying Windows",
      value: optimalWindows,
      subtext: "optimal now",
      icon: Target,
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
    },
    {
      label: "Supply Risk",
      value: highRisk,
      subtext: "elevated",
      icon: AlertTriangle,
      color: "text-amber-400",
      bgColor: "bg-amber-500/10",
    },
    {
      label: "Avg Savings",
      value: `${avgSavings.toFixed(1)}%`,
      subtext: "potential",
      icon: PiggyBank,
      color: "text-cyan-400",
      bgColor: "bg-cyan-500/10",
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      {stats.map((stat) => (
        <Card key={stat.label} className="bg-neutral-900/50 border-neutral-800">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className={cn("p-2 rounded-lg", stat.bgColor)}>
                <stat.icon className={cn("w-5 h-5", stat.color)} />
              </div>
              <div>
                <div className="text-2xl font-bold text-neutral-100">{stat.value}</div>
                <div className="text-xs text-neutral-500">{stat.subtext}</div>
              </div>
            </div>
            <div className="text-xs text-neutral-400 mt-2">{stat.label}</div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

// Priority Actions Panel
function PriorityActions({ opportunities }: { opportunities: BuyingOpportunity[] }) {
  if (opportunities.length === 0) {
    return (
      <Card className="bg-neutral-900/50 border-neutral-800 mb-6">
        <CardContent className="p-6 text-center">
          <CheckCircle2 className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
          <p className="text-neutral-300 font-medium">No Urgent Actions Required</p>
          <p className="text-sm text-neutral-500">Market conditions are stable for purchasing</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-gradient-to-br from-emerald-500/5 via-teal-500/5 to-emerald-500/5 border-emerald-500/20 mb-6">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Zap className="w-5 h-5 text-emerald-400" />
          Priority Buying Opportunities
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {opportunities.map((opp) => (
            <div
              key={opp.assetId}
              className={cn(
                "p-4 rounded-xl border transition-all",
                opp.urgency === "buy-now"
                  ? "bg-red-500/5 border-red-500/20"
                  : "bg-emerald-500/5 border-emerald-500/20"
              )}
            >
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h4 className="font-semibold text-neutral-100">{opp.name}</h4>
                  <span className="text-sm text-neutral-400">
                    ${opp.currentPrice.toLocaleString()}{opp.unit}
                  </span>
                </div>
                <Badge
                  className={cn(
                    opp.urgency === "buy-now"
                      ? "bg-red-500/20 text-red-400 border-red-500/30"
                      : "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"
                  )}
                >
                  {opp.urgency === "buy-now" ? "ACT NOW" : "OPPORTUNITY"}
                </Badge>
              </div>

              <p className="text-sm text-neutral-400 mb-3">{opp.reasoning}</p>

              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-4">
                  <span className="text-neutral-500">
                    <Clock className="w-3.5 h-3.5 inline mr-1" />
                    {opp.timeframe}
                  </span>
                  <span className="text-neutral-500">
                    <Target className="w-3.5 h-3.5 inline mr-1" />
                    {opp.confidence}% conf.
                  </span>
                </div>
                {opp.urgency !== "buy-now" && (
                  <span className="text-emerald-400 font-medium">
                    Save up to {opp.savingsPotential}%
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// Commodity Card for Procurement
function CommodityCard({ commodity }: { commodity: CommodityData }) {
  const { procurement } = commodity;

  const windowConfig = {
    optimal: {
      color: "text-emerald-400",
      bg: "bg-emerald-500/10",
      border: "border-emerald-500/20",
      label: "Optimal Window",
      icon: CheckCircle2,
    },
    urgent: {
      color: "text-red-400",
      bg: "bg-red-500/10",
      border: "border-red-500/20",
      label: "Buy Now",
      icon: AlertTriangle,
    },
    consider: {
      color: "text-blue-400",
      bg: "bg-blue-500/10",
      border: "border-blue-500/20",
      label: "Consider",
      icon: Target,
    },
    acceptable: {
      color: "text-neutral-400",
      bg: "bg-neutral-500/10",
      border: "border-neutral-500/20",
      label: "Standard",
      icon: CircleDot,
    },
    wait: {
      color: "text-amber-400",
      bg: "bg-amber-500/10",
      border: "border-amber-500/20",
      label: "Wait",
      icon: Clock,
    },
  };

  const config = windowConfig[procurement.buyWindow];
  const WindowIcon = config.icon;

  const outlookConfig = {
    rising: { icon: TrendingUp, color: "text-red-400", label: "Rising" },
    falling: { icon: TrendingDown, color: "text-emerald-400", label: "Falling" },
    stable: { icon: Minus, color: "text-neutral-400", label: "Stable" },
  };

  const outlook = outlookConfig[procurement.priceOutlook];
  const OutlookIcon = outlook.icon;

  return (
    <Card className={cn("border transition-all hover:border-neutral-600", config.bg, config.border)}>
      <CardContent className="p-4">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="font-semibold text-neutral-100">{commodity.name}</h3>
            <div className="flex items-center gap-2 text-sm text-neutral-400">
              <span>{commodity.symbol}</span>
              <span>•</span>
              <span className={commodity.changePercent24h >= 0 ? "text-emerald-400" : "text-red-400"}>
                {commodity.changePercent24h >= 0 ? "+" : ""}
                {commodity.changePercent24h.toFixed(2)}%
              </span>
            </div>
          </div>
          <div className="text-right">
            <div className="text-lg font-bold text-neutral-100">
              ${commodity.currentPrice.toLocaleString()}
            </div>
            <div className="text-xs text-neutral-500">{commodity.unit}</div>
          </div>
        </div>

        {/* Buy Window Badge */}
        <div className={cn("flex items-center gap-2 p-2 rounded-lg mb-4", config.bg)}>
          <WindowIcon className={cn("w-4 h-4", config.color)} />
          <span className={cn("text-sm font-medium", config.color)}>{config.label}</span>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="p-2 bg-neutral-800/50 rounded-lg">
            <div className="flex items-center gap-1.5 mb-1">
              <OutlookIcon className={cn("w-3.5 h-3.5", outlook.color)} />
              <span className="text-xs text-neutral-500">Price Outlook</span>
            </div>
            <span className={cn("text-sm font-medium", outlook.color)}>{outlook.label}</span>
          </div>
          <div className="p-2 bg-neutral-800/50 rounded-lg">
            <div className="flex items-center gap-1.5 mb-1">
              <Shield className="w-3.5 h-3.5 text-neutral-400" />
              <span className="text-xs text-neutral-500">Supply Risk</span>
            </div>
            <span
              className={cn(
                "text-sm font-medium capitalize",
                procurement.supplyRisk === "high"
                  ? "text-red-400"
                  : procurement.supplyRisk === "medium"
                  ? "text-amber-400"
                  : "text-emerald-400"
              )}
            >
              {procurement.supplyRisk}
            </span>
          </div>
        </div>

        {/* Recommendation */}
        <div className="p-3 bg-neutral-800/30 rounded-lg">
          <div className="flex items-start gap-2">
            <FileText className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <div>
              <span className="text-xs text-neutral-500">Recommendation</span>
              <p className="text-sm text-neutral-300 mt-0.5">{procurement.recommendedAction}</p>
            </div>
          </div>
        </div>

        {/* Savings Badge */}
        {procurement.savingsOpportunity > 0 && (
          <div className="mt-3 flex items-center justify-between">
            <span className="text-xs text-neutral-500">Potential Savings</span>
            <Badge className="bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
              <PiggyBank className="w-3 h-3 mr-1" />
              Up to {procurement.savingsOpportunity}%
            </Badge>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Contract Timing Guidance
function ContractGuidance({ commodities }: { commodities: CommodityData[] }) {
  const shortTermBuys = commodities.filter((c) => c.procurement.priceOutlook === "falling");
  const lockInNow = commodities.filter((c) => c.procurement.priceOutlook === "rising");

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Calendar className="w-5 h-5 text-emerald-400" />
          Contract Timing Guidance
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0 space-y-4">
        {lockInNow.length > 0 && (
          <div className="p-3 bg-red-500/5 border border-red-500/20 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-red-400" />
              <span className="text-sm font-medium text-red-400">Lock In Long-Term</span>
            </div>
            <p className="text-xs text-neutral-400 mb-2">
              Consider longer-term contracts for these commodities before prices rise:
            </p>
            <div className="flex flex-wrap gap-2">
              {lockInNow.map((c) => (
                <Badge key={c.assetId} variant="outline" className="text-neutral-300 border-neutral-600">
                  {c.symbol}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {shortTermBuys.length > 0 && (
          <div className="p-3 bg-emerald-500/5 border border-emerald-500/20 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <TrendingDown className="w-4 h-4 text-emerald-400" />
              <span className="text-sm font-medium text-emerald-400">Short-Term Contracts</span>
            </div>
            <p className="text-xs text-neutral-400 mb-2">
              Use short-term contracts - better prices may be ahead:
            </p>
            <div className="flex flex-wrap gap-2">
              {shortTermBuys.map((c) => (
                <Badge key={c.assetId} variant="outline" className="text-neutral-300 border-neutral-600">
                  {c.symbol}
                </Badge>
              ))}
            </div>
          </div>
        )}

        <div className="p-3 bg-neutral-800/50 rounded-lg">
          <div className="flex items-start gap-2">
            <Calculator className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
            <div className="text-xs text-neutral-400">
              <strong className="text-neutral-300">Pro Tip:</strong> Review contract terms quarterly
              and align procurement cycles with seasonal patterns for maximum savings.
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// Supply Chain Risk Monitor
function SupplyRiskMonitor({ commodities }: { commodities: CommodityData[] }) {
  const highRisk = commodities.filter((c) => c.procurement.supplyRisk === "high");
  const mediumRisk = commodities.filter((c) => c.procurement.supplyRisk === "medium");

  return (
    <Card className="bg-neutral-900/50 border-neutral-800">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Globe className="w-5 h-5 text-amber-400" />
          Supply Chain Risk Monitor
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0 space-y-3">
        {highRisk.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span className="text-sm text-red-400 font-medium">Elevated Risk</span>
            </div>
            {highRisk.map((c) => (
              <div key={c.assetId} className="flex items-center justify-between p-2 bg-red-500/5 rounded-lg">
                <span className="text-sm text-neutral-300">{c.name}</span>
                <span className="text-xs text-neutral-500">Monitor closely</span>
              </div>
            ))}
          </div>
        )}

        {mediumRisk.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-amber-500" />
              <span className="text-sm text-amber-400 font-medium">Moderate Risk</span>
            </div>
            {mediumRisk.map((c) => (
              <div key={c.assetId} className="flex items-center justify-between p-2 bg-amber-500/5 rounded-lg">
                <span className="text-sm text-neutral-300">{c.name}</span>
                <span className="text-xs text-neutral-500">Standard monitoring</span>
              </div>
            ))}
          </div>
        )}

        {highRisk.length === 0 && mediumRisk.length === 0 && (
          <div className="p-4 text-center">
            <CheckCircle2 className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
            <p className="text-sm text-neutral-400">All supply chains operating normally</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// Category Filter
function CategoryFilter({
  selected,
  onChange,
}: {
  selected: string;
  onChange: (category: string) => void;
}) {
  const categories = [
    { id: "all", label: "All", icon: Layers },
    { id: "energy", label: "Energy", icon: Zap },
    { id: "metals", label: "Metals", icon: Factory },
    { id: "agriculture", label: "Agriculture", icon: Warehouse },
  ];

  return (
    <div className="flex items-center gap-2 mb-4">
      {categories.map((cat) => (
        <button
          key={cat.id}
          onClick={() => onChange(cat.id)}
          className={cn(
            "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-all",
            selected === cat.id
              ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
              : "bg-neutral-800/50 text-neutral-400 hover:text-neutral-200 border border-transparent"
          )}
        >
          <cat.icon className="w-4 h-4" />
          {cat.label}
        </button>
      ))}
    </div>
  );
}

// ============================================================================
// Constants
// ============================================================================

const ASSET_CATEGORIES: Record<string, string> = {
  "crude-oil": "energy",
  "natural-gas": "energy",
  gold: "metals",
  silver: "metals",
  copper: "metals",
  platinum: "metals",
  wheat: "agriculture",
  corn: "agriculture",
  soybean: "agriculture",
};

// ============================================================================
// Main Dashboard
// ============================================================================

export function ProcurementDashboard() {
  const commodities = useMemo(() => getCommodityData(), []);
  const opportunities = useMemo(() => getBuyingOpportunities(commodities), [commodities]);
  const [categoryFilter, setCategoryFilter] = useState("all");

  const filteredCommodities = useMemo(() => {
    if (categoryFilter === "all") return commodities;
    return commodities.filter((c) => ASSET_CATEGORIES[c.assetId] === categoryFilter);
  }, [commodities, categoryFilter]);

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <DashboardHeader />

      {/* Market Overview */}
      <MarketOverview commodities={commodities} />

      {/* Priority Actions */}
      <PriorityActions opportunities={opportunities} />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Commodity Cards - Main Content */}
        <div className="lg:col-span-3">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-neutral-200 flex items-center gap-2">
              <Package className="w-5 h-5 text-emerald-400" />
              Commodity Procurement Status
            </h2>
          </div>

          <CategoryFilter selected={categoryFilter} onChange={setCategoryFilter} />

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {filteredCommodities.map((commodity) => (
              <CommodityCard key={commodity.assetId} commodity={commodity} />
            ))}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-4">
          <ContractGuidance commodities={commodities} />
          <SupplyRiskMonitor commodities={commodities} />

          {/* Procurement Tips */}
          <Card className="bg-emerald-500/5 border-emerald-500/20">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <Scale className="w-5 h-5 text-emerald-400 shrink-0" />
                <div className="text-xs text-neutral-400">
                  <strong className="text-emerald-400">Smart Procurement:</strong> AI signals help
                  time your purchases. Combine with supplier relationships and inventory levels for
                  best results.
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
