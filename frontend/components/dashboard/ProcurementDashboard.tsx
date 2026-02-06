"use client";

import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { MOCK_ASSETS, MOCK_SIGNALS } from "@/lib/mock-data";
import type { AssetId } from "@/types";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  Clock,
  Calendar,
  Globe,
  Shield,
  Activity,
  FileText,
  Calculator,
  DollarSign,
  Building2,
  ClipboardCheck,
  AlertCircle,
  BarChart3,
  PieChart,
  ArrowUpRight,
  ArrowDownRight,
  CheckSquare,
  Square,
  Users,
  Truck,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

interface CommodityForecast {
  assetId: AssetId;
  name: string;
  symbol: string;
  currentPrice: number;
  unit: string;
  forecast: {
    point: number;
    low: number;
    high: number;
    confidence: number;
  };
  change: number;
  direction: "up" | "down" | "flat";
}

interface CostBasis {
  commodity: string;
  avgCost: number;
  currentPrice: number;
  variance: number;
  ytdSpend: number;
  ytdVolume: number;
}

interface SupplierExposure {
  supplier: string;
  region: string;
  commodities: string[];
  riskScore: number;
  spend: number;
  contracts: number;
}

interface ComplianceItem {
  id: string;
  category: string;
  item: string;
  status: "complete" | "pending" | "overdue" | "na";
  dueDate?: string;
  assignee?: string;
}

interface BudgetItem {
  category: string;
  budget: number;
  actual: number;
  variance: number;
  trend: "up" | "down" | "flat";
}

interface ContractExpiration {
  id: string;
  supplier: string;
  commodity: string;
  expiryDate: string;
  value: number;
  status: "active" | "expiring" | "expired" | "renewal";
}

// ============================================================================
// Mock Data
// ============================================================================

const COMMODITY_UNITS: Record<string, string> = {
  "crude-oil": "/bbl",
  "natural-gas": "/MMBtu",
  gold: "/oz",
  silver: "/oz",
  copper: "/lb",
  platinum: "/oz",
  wheat: "/bu",
  corn: "/bu",
  soybean: "/bu",
};

function generateForecasts(): CommodityForecast[] {
  const forecasts: CommodityForecast[] = [];
  const procurementAssets = Object.entries(MOCK_ASSETS).filter(
    ([id]) => id !== "bitcoin"
  );

  procurementAssets.forEach(([assetId, asset]) => {
    const signal = MOCK_SIGNALS[assetId as AssetId]?.["D+5"];
    if (signal) {
      const volatility = 0.05 + Math.random() * 0.1;
      const direction = signal.direction === "bullish" ? "up" : signal.direction === "bearish" ? "down" : "flat";
      const change = direction === "up" ? Math.random() * 8 : direction === "down" ? -Math.random() * 8 : (Math.random() - 0.5) * 3;
      const point = asset.currentPrice * (1 + change / 100);

      forecasts.push({
        assetId: assetId as AssetId,
        name: asset.name,
        symbol: asset.symbol,
        currentPrice: asset.currentPrice,
        unit: COMMODITY_UNITS[assetId] || "",
        forecast: {
          point: Number(point.toFixed(2)),
          low: Number((point * (1 - volatility)).toFixed(2)),
          high: Number((point * (1 + volatility)).toFixed(2)),
          confidence: signal.confidence,
        },
        change: Number(change.toFixed(2)),
        direction,
      });
    }
  });

  return forecasts;
}

function generateCostBasis(): CostBasis[] {
  return [
    { commodity: "Crude Oil", avgCost: 71.25, currentPrice: 73.45, variance: 3.1, ytdSpend: 12400000, ytdVolume: 173850 },
    { commodity: "Natural Gas", avgCost: 2.85, currentPrice: 2.72, variance: -4.6, ytdSpend: 4200000, ytdVolume: 1545455 },
    { commodity: "Gold", avgCost: 2285.00, currentPrice: 2312.50, variance: 1.2, ytdSpend: 8900000, ytdVolume: 3850 },
    { commodity: "Copper", avgCost: 4.15, currentPrice: 4.08, variance: -1.7, ytdSpend: 6100000, ytdVolume: 1493902 },
    { commodity: "Wheat", avgCost: 5.82, currentPrice: 5.95, variance: 2.2, ytdSpend: 2800000, ytdVolume: 470790 },
  ];
}

function generateSupplierExposure(): SupplierExposure[] {
  return [
    { supplier: "GlobalPetro Inc", region: "Middle East", commodities: ["Crude Oil"], riskScore: 65, spend: 8500000, contracts: 4 },
    { supplier: "MetalWorks Ltd", region: "South America", commodities: ["Copper", "Silver"], riskScore: 42, spend: 4200000, contracts: 3 },
    { supplier: "EuroGas GmbH", region: "Europe", commodities: ["Natural Gas"], riskScore: 28, spend: 3100000, contracts: 2 },
    { supplier: "Pacific Grains", region: "Asia Pacific", commodities: ["Wheat", "Corn", "Soybean"], riskScore: 35, spend: 2800000, contracts: 5 },
    { supplier: "NorthAm Energy", region: "North America", commodities: ["Crude Oil", "Natural Gas"], riskScore: 22, spend: 5600000, contracts: 3 },
    { supplier: "Precious Metals Co", region: "Africa", commodities: ["Gold", "Platinum"], riskScore: 55, spend: 6200000, contracts: 2 },
  ];
}

function generateComplianceItems(): ComplianceItem[] {
  return [
    { id: "1", category: "Supplier Audit", item: "Annual supplier compliance audit - GlobalPetro", status: "complete", assignee: "J. Smith" },
    { id: "2", category: "Documentation", item: "Update commodity sourcing policy", status: "pending", dueDate: "2024-02-15", assignee: "M. Johnson" },
    { id: "3", category: "Certification", item: "ISO 14001 recertification", status: "pending", dueDate: "2024-03-01", assignee: "R. Williams" },
    { id: "4", category: "Reporting", item: "Q4 sustainability report submission", status: "overdue", dueDate: "2024-01-31", assignee: "A. Davis" },
    { id: "5", category: "Training", item: "Anti-bribery compliance training", status: "complete", assignee: "All Staff" },
    { id: "6", category: "Contract Review", item: "Force majeure clause review", status: "pending", dueDate: "2024-02-28", assignee: "Legal Team" },
    { id: "7", category: "Risk Assessment", item: "Quarterly supplier risk assessment", status: "complete", assignee: "K. Brown" },
    { id: "8", category: "ESG", item: "Scope 3 emissions baseline", status: "pending", dueDate: "2024-04-15", assignee: "Sustainability" },
  ];
}

function generateBudgetItems(): BudgetItem[] {
  return [
    { category: "Energy", budget: 18000000, actual: 16700000, variance: -7.2, trend: "down" },
    { category: "Precious Metals", budget: 12000000, actual: 15100000, variance: 25.8, trend: "up" },
    { category: "Base Metals", budget: 8000000, actual: 6100000, variance: -23.8, trend: "down" },
    { category: "Agriculture", budget: 5000000, actual: 5300000, variance: 6.0, trend: "up" },
    { category: "Logistics", budget: 3500000, actual: 3200000, variance: -8.6, trend: "down" },
  ];
}

function generateContractExpirations(): ContractExpiration[] {
  return [
    { id: "1", supplier: "GlobalPetro Inc", commodity: "Crude Oil", expiryDate: "2024-02-28", value: 4200000, status: "expiring" },
    { id: "2", supplier: "EuroGas GmbH", commodity: "Natural Gas", expiryDate: "2024-04-15", value: 1800000, status: "active" },
    { id: "3", supplier: "MetalWorks Ltd", commodity: "Copper", expiryDate: "2024-01-31", value: 2100000, status: "renewal" },
    { id: "4", supplier: "Pacific Grains", commodity: "Wheat", expiryDate: "2024-06-30", value: 950000, status: "active" },
    { id: "5", supplier: "NorthAm Energy", commodity: "Natural Gas", expiryDate: "2024-03-15", value: 2800000, status: "expiring" },
    { id: "6", supplier: "Precious Metals Co", commodity: "Gold", expiryDate: "2024-05-20", value: 3500000, status: "active" },
  ];
}

// ============================================================================
// Enterprise Light Theme Wrapper
// ============================================================================

function LightModeWrapper({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen -m-6 bg-slate-50 text-slate-900">
      {children}
    </div>
  );
}

// ============================================================================
// Header Component
// ============================================================================

function DashboardHeader() {
  const today = new Date().toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  return (
    <div className="bg-white border-b border-slate-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="p-2.5 bg-blue-600 rounded-lg">
            <Building2 className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-slate-900">Procurement Dashboard</h1>
            <p className="text-sm text-slate-500">{today}</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Badge className="bg-green-100 text-green-700 border-green-200 hover:bg-green-100">
            <Activity className="w-3.5 h-3.5 mr-1.5" />
            Markets Open
          </Badge>
          <Badge className="bg-blue-100 text-blue-700 border-blue-200 hover:bg-blue-100">
            <Clock className="w-3.5 h-3.5 mr-1.5" />
            Last Updated: {new Date().toLocaleTimeString()}
          </Badge>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Summary Cards
// ============================================================================

function SummaryCards() {
  const cards = [
    {
      title: "Total YTD Spend",
      value: "$34.4M",
      change: "+8.2%",
      trend: "up" as const,
      icon: DollarSign,
      color: "blue",
    },
    {
      title: "Active Contracts",
      value: "19",
      change: "3 expiring",
      trend: "warning" as const,
      icon: FileText,
      color: "amber",
    },
    {
      title: "Supplier Risk Score",
      value: "Low",
      change: "Avg: 38/100",
      trend: "down" as const,
      icon: Shield,
      color: "green",
    },
    {
      title: "Compliance Status",
      value: "87%",
      change: "1 overdue",
      trend: "warning" as const,
      icon: ClipboardCheck,
      color: "purple",
    },
  ];

  const colorMap = {
    blue: { bg: "bg-blue-50", icon: "text-blue-600", border: "border-blue-200" },
    amber: { bg: "bg-amber-50", icon: "text-amber-600", border: "border-amber-200" },
    green: { bg: "bg-green-50", icon: "text-green-600", border: "border-green-200" },
    purple: { bg: "bg-purple-50", icon: "text-purple-600", border: "border-purple-200" },
  };

  return (
    <div className="grid grid-cols-4 gap-4 p-6">
      {cards.map((card) => {
        const colors = colorMap[card.color as keyof typeof colorMap];
        return (
          <Card key={card.title} className={cn("bg-white border shadow-sm", colors.border)}>
            <CardContent className="p-4">
              <div className="flex items-start justify-between">
                <div>
                  <p className="text-sm text-slate-500 mb-1">{card.title}</p>
                  <p className="text-2xl font-semibold text-slate-900">{card.value}</p>
                  <div className="flex items-center gap-1 mt-1">
                    {card.trend === "up" && <ArrowUpRight className="w-3.5 h-3.5 text-green-600" />}
                    {card.trend === "down" && <ArrowDownRight className="w-3.5 h-3.5 text-green-600" />}
                    {card.trend === "warning" && <AlertCircle className="w-3.5 h-3.5 text-amber-600" />}
                    <span className={cn(
                      "text-xs",
                      card.trend === "warning" ? "text-amber-600" : "text-green-600"
                    )}>
                      {card.change}
                    </span>
                  </div>
                </div>
                <div className={cn("p-2 rounded-lg", colors.bg)}>
                  <card.icon className={cn("w-5 h-5", colors.icon)} />
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

// ============================================================================
// Price Forecasts with Confidence Intervals
// ============================================================================

function PriceForecastPanel({ forecasts }: { forecasts: CommodityForecast[] }) {
  return (
    <Card className="bg-white border border-slate-200 shadow-sm">
      <CardHeader className="border-b border-slate-100 pb-3">
        <CardTitle className="text-base font-semibold text-slate-900 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-blue-600" />
          5-Day Price Forecasts
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="divide-y divide-slate-100">
          {forecasts.slice(0, 6).map((item) => (
            <div key={item.assetId} className="p-4 hover:bg-slate-50 transition-colors">
              <div className="flex items-center justify-between mb-2">
                <div>
                  <span className="font-medium text-slate-900">{item.name}</span>
                  <span className="text-slate-400 text-sm ml-2">{item.symbol}</span>
                </div>
                <div className="text-right">
                  <span className="font-semibold text-slate-900">
                    ${item.currentPrice.toLocaleString()}{item.unit}
                  </span>
                </div>
              </div>

              {/* Confidence Interval Visualization */}
              <div className="relative h-8 bg-slate-100 rounded-lg overflow-hidden">
                {/* Range bar */}
                <div
                  className="absolute h-full bg-blue-100 rounded"
                  style={{
                    left: `${((item.forecast.low - item.currentPrice * 0.85) / (item.currentPrice * 0.3)) * 100}%`,
                    width: `${((item.forecast.high - item.forecast.low) / (item.currentPrice * 0.3)) * 100}%`,
                  }}
                />
                {/* Point estimate marker */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-blue-600"
                  style={{
                    left: `${((item.forecast.point - item.currentPrice * 0.85) / (item.currentPrice * 0.3)) * 100}%`,
                  }}
                />
                {/* Current price marker */}
                <div
                  className="absolute top-0 bottom-0 w-0.5 bg-slate-400"
                  style={{
                    left: `${((item.currentPrice - item.currentPrice * 0.85) / (item.currentPrice * 0.3)) * 100}%`,
                  }}
                />
              </div>

              <div className="flex items-center justify-between mt-2 text-xs">
                <div className="flex items-center gap-3">
                  <span className="text-slate-500">
                    Low: <span className="text-slate-700">${item.forecast.low.toLocaleString()}</span>
                  </span>
                  <span className="text-blue-600 font-medium">
                    Est: ${item.forecast.point.toLocaleString()}
                  </span>
                  <span className="text-slate-500">
                    High: <span className="text-slate-700">${item.forecast.high.toLocaleString()}</span>
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {item.direction === "up" && (
                    <span className="text-red-600 flex items-center">
                      <TrendingUp className="w-3.5 h-3.5 mr-1" />
                      +{item.change}%
                    </span>
                  )}
                  {item.direction === "down" && (
                    <span className="text-green-600 flex items-center">
                      <TrendingDown className="w-3.5 h-3.5 mr-1" />
                      {item.change}%
                    </span>
                  )}
                  {item.direction === "flat" && (
                    <span className="text-slate-500 flex items-center">
                      <Minus className="w-3.5 h-3.5 mr-1" />
                      {item.change}%
                    </span>
                  )}
                  <Badge className="bg-slate-100 text-slate-600 border-slate-200 text-[10px]">
                    {item.forecast.confidence}% conf
                  </Badge>
                </div>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Cost Basis Analysis
// ============================================================================

function CostBasisPanel({ data }: { data: CostBasis[] }) {
  return (
    <Card className="bg-white border border-slate-200 shadow-sm">
      <CardHeader className="border-b border-slate-100 pb-3">
        <CardTitle className="text-base font-semibold text-slate-900 flex items-center gap-2">
          <Calculator className="w-5 h-5 text-purple-600" />
          Cost Basis Analysis
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <table className="w-full">
          <thead className="bg-slate-50">
            <tr className="text-xs text-slate-500 uppercase tracking-wider">
              <th className="text-left p-3 font-medium">Commodity</th>
              <th className="text-right p-3 font-medium">Avg Cost</th>
              <th className="text-right p-3 font-medium">Current</th>
              <th className="text-right p-3 font-medium">Variance</th>
              <th className="text-right p-3 font-medium">YTD Spend</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {data.map((item) => (
              <tr key={item.commodity} className="hover:bg-slate-50 transition-colors">
                <td className="p-3 font-medium text-slate-900">{item.commodity}</td>
                <td className="p-3 text-right text-slate-600">${item.avgCost.toFixed(2)}</td>
                <td className="p-3 text-right text-slate-900">${item.currentPrice.toFixed(2)}</td>
                <td className={cn(
                  "p-3 text-right font-medium",
                  item.variance >= 0 ? "text-red-600" : "text-green-600"
                )}>
                  {item.variance >= 0 ? "+" : ""}{item.variance.toFixed(1)}%
                </td>
                <td className="p-3 text-right text-slate-600">
                  ${(item.ytdSpend / 1000000).toFixed(1)}M
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Supplier Exposure Heatmap
// ============================================================================

function SupplierExposureHeatmap({ data }: { data: SupplierExposure[] }) {
  const getRiskColor = (score: number) => {
    if (score >= 60) return "bg-red-100 text-red-700 border-red-200";
    if (score >= 40) return "bg-amber-100 text-amber-700 border-amber-200";
    return "bg-green-100 text-green-700 border-green-200";
  };

  const getRiskBg = (score: number) => {
    if (score >= 60) return "bg-red-500";
    if (score >= 40) return "bg-amber-500";
    return "bg-green-500";
  };

  return (
    <Card className="bg-white border border-slate-200 shadow-sm">
      <CardHeader className="border-b border-slate-100 pb-3">
        <CardTitle className="text-base font-semibold text-slate-900 flex items-center gap-2">
          <Globe className="w-5 h-5 text-amber-600" />
          Supplier Exposure & Risk
        </CardTitle>
      </CardHeader>
      <CardContent className="p-4">
        <div className="grid grid-cols-2 gap-3">
          {data.map((supplier) => (
            <div
              key={supplier.supplier}
              className="p-3 border border-slate-200 rounded-lg hover:border-slate-300 transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                <div>
                  <p className="font-medium text-slate-900 text-sm">{supplier.supplier}</p>
                  <p className="text-xs text-slate-500">{supplier.region}</p>
                </div>
                <Badge className={cn("text-[10px]", getRiskColor(supplier.riskScore))}>
                  Risk: {supplier.riskScore}
                </Badge>
              </div>

              {/* Risk bar */}
              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden mb-2">
                <div
                  className={cn("h-full rounded-full transition-all", getRiskBg(supplier.riskScore))}
                  style={{ width: `${supplier.riskScore}%` }}
                />
              </div>

              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-500">
                  {supplier.commodities.length} commodities
                </span>
                <span className="text-slate-700">
                  ${(supplier.spend / 1000000).toFixed(1)}M spend
                </span>
              </div>
            </div>
          ))}
        </div>

        {/* Legend */}
        <div className="flex items-center justify-center gap-6 mt-4 pt-4 border-t border-slate-100">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span className="text-xs text-slate-500">Low Risk (0-39)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-amber-500" />
            <span className="text-xs text-slate-500">Medium (40-59)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span className="text-xs text-slate-500">High Risk (60+)</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Compliance Checklist
// ============================================================================

function ComplianceChecklist({ items }: { items: ComplianceItem[] }) {
  const complete = items.filter((i) => i.status === "complete").length;
  const pending = items.filter((i) => i.status === "pending").length;
  const overdue = items.filter((i) => i.status === "overdue").length;

  const getStatusBadge = (status: ComplianceItem["status"]) => {
    switch (status) {
      case "complete":
        return <Badge className="bg-green-100 text-green-700 border-green-200 text-[10px]">Complete</Badge>;
      case "pending":
        return <Badge className="bg-blue-100 text-blue-700 border-blue-200 text-[10px]">Pending</Badge>;
      case "overdue":
        return <Badge className="bg-red-100 text-red-700 border-red-200 text-[10px]">Overdue</Badge>;
      default:
        return <Badge className="bg-slate-100 text-slate-600 border-slate-200 text-[10px]">N/A</Badge>;
    }
  };

  const getStatusIcon = (status: ComplianceItem["status"]) => {
    switch (status) {
      case "complete":
        return <CheckSquare className="w-4 h-4 text-green-600" />;
      case "overdue":
        return <AlertTriangle className="w-4 h-4 text-red-600" />;
      default:
        return <Square className="w-4 h-4 text-slate-400" />;
    }
  };

  return (
    <Card className="bg-white border border-slate-200 shadow-sm">
      <CardHeader className="border-b border-slate-100 pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-semibold text-slate-900 flex items-center gap-2">
            <ClipboardCheck className="w-5 h-5 text-green-600" />
            Compliance Status
          </CardTitle>
          <div className="flex items-center gap-2 text-xs">
            <span className="text-green-600">{complete} complete</span>
            <span className="text-slate-300">|</span>
            <span className="text-blue-600">{pending} pending</span>
            <span className="text-slate-300">|</span>
            <span className="text-red-600">{overdue} overdue</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-0 max-h-80 overflow-y-auto">
        <div className="divide-y divide-slate-100">
          {items.map((item) => (
            <div
              key={item.id}
              className={cn(
                "p-3 flex items-start gap-3 hover:bg-slate-50 transition-colors",
                item.status === "overdue" && "bg-red-50/50"
              )}
            >
              {getStatusIcon(item.status)}
              <div className="flex-1 min-w-0">
                <p className="text-sm text-slate-900 truncate">{item.item}</p>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-xs text-slate-500">{item.category}</span>
                  {item.dueDate && (
                    <>
                      <span className="text-slate-300">•</span>
                      <span className={cn(
                        "text-xs",
                        item.status === "overdue" ? "text-red-600" : "text-slate-500"
                      )}>
                        Due: {item.dueDate}
                      </span>
                    </>
                  )}
                  {item.assignee && (
                    <>
                      <span className="text-slate-300">•</span>
                      <span className="text-xs text-slate-500">{item.assignee}</span>
                    </>
                  )}
                </div>
              </div>
              {getStatusBadge(item.status)}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Budget vs Actual
// ============================================================================

function BudgetVsActualPanel({ data }: { data: BudgetItem[] }) {
  const totalBudget = data.reduce((sum, item) => sum + item.budget, 0);
  const totalActual = data.reduce((sum, item) => sum + item.actual, 0);
  const totalVariance = ((totalActual - totalBudget) / totalBudget) * 100;

  return (
    <Card className="bg-white border border-slate-200 shadow-sm">
      <CardHeader className="border-b border-slate-100 pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base font-semibold text-slate-900 flex items-center gap-2">
            <PieChart className="w-5 h-5 text-blue-600" />
            Budget vs Actual (YTD)
          </CardTitle>
          <Badge className={cn(
            "text-xs",
            totalVariance > 0 ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"
          )}>
            {totalVariance > 0 ? "+" : ""}{totalVariance.toFixed(1)}% variance
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="p-4">
        <div className="space-y-4">
          {data.map((item) => (
            <div key={item.category}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium text-slate-900">{item.category}</span>
                <div className="flex items-center gap-3 text-xs">
                  <span className="text-slate-500">
                    Budget: ${(item.budget / 1000000).toFixed(1)}M
                  </span>
                  <span className={cn(
                    "font-medium",
                    item.variance > 0 ? "text-red-600" : "text-green-600"
                  )}>
                    {item.variance > 0 ? "+" : ""}{item.variance.toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="relative h-6 bg-slate-100 rounded overflow-hidden">
                {/* Budget line */}
                <div className="absolute inset-0 border-r-2 border-slate-400 border-dashed" style={{ width: "100%" }} />
                {/* Actual bar */}
                <div
                  className={cn(
                    "h-full rounded transition-all",
                    item.variance > 10 ? "bg-red-500" : item.variance > 0 ? "bg-amber-500" : "bg-green-500"
                  )}
                  style={{ width: `${Math.min((item.actual / item.budget) * 100, 150)}%` }}
                />
                {/* Actual value label */}
                <div className="absolute inset-0 flex items-center px-2">
                  <span className="text-xs font-medium text-white drop-shadow">
                    ${(item.actual / 1000000).toFixed(1)}M
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Totals */}
        <div className="mt-4 pt-4 border-t border-slate-200">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-slate-900">Total</span>
            <div className="text-right">
              <p className="text-lg font-semibold text-slate-900">
                ${(totalActual / 1000000).toFixed(1)}M
                <span className="text-sm text-slate-500 font-normal ml-2">
                  of ${(totalBudget / 1000000).toFixed(1)}M
                </span>
              </p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Contract Expiration Calendar
// ============================================================================

function ContractCalendar({ contracts }: { contracts: ContractExpiration[] }) {
  const sortedContracts = [...contracts].sort(
    (a, b) => new Date(a.expiryDate).getTime() - new Date(b.expiryDate).getTime()
  );

  const getStatusStyle = (status: ContractExpiration["status"]) => {
    switch (status) {
      case "expiring":
        return "bg-amber-50 border-amber-200 text-amber-700";
      case "expired":
        return "bg-red-50 border-red-200 text-red-700";
      case "renewal":
        return "bg-blue-50 border-blue-200 text-blue-700";
      default:
        return "bg-green-50 border-green-200 text-green-700";
    }
  };

  const getStatusLabel = (status: ContractExpiration["status"]) => {
    switch (status) {
      case "expiring":
        return "Expiring Soon";
      case "expired":
        return "Expired";
      case "renewal":
        return "In Renewal";
      default:
        return "Active";
    }
  };

  return (
    <Card className="bg-white border border-slate-200 shadow-sm">
      <CardHeader className="border-b border-slate-100 pb-3">
        <CardTitle className="text-base font-semibold text-slate-900 flex items-center gap-2">
          <Calendar className="w-5 h-5 text-indigo-600" />
          Contract Expiration Calendar
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div className="divide-y divide-slate-100">
          {sortedContracts.map((contract) => (
            <div
              key={contract.id}
              className={cn(
                "p-4 flex items-center justify-between hover:bg-slate-50 transition-colors",
                contract.status === "expiring" && "bg-amber-50/30",
                contract.status === "renewal" && "bg-blue-50/30"
              )}
            >
              <div className="flex items-center gap-4">
                <div className="text-center min-w-[60px]">
                  <p className="text-2xl font-bold text-slate-900">
                    {new Date(contract.expiryDate).getDate()}
                  </p>
                  <p className="text-xs text-slate-500 uppercase">
                    {new Date(contract.expiryDate).toLocaleDateString("en-US", { month: "short" })}
                  </p>
                </div>
                <div>
                  <p className="font-medium text-slate-900">{contract.supplier}</p>
                  <p className="text-sm text-slate-500">{contract.commodity}</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <p className="font-semibold text-slate-900">
                    ${(contract.value / 1000000).toFixed(1)}M
                  </p>
                  <p className="text-xs text-slate-500">Contract Value</p>
                </div>
                <Badge className={cn("text-xs min-w-[90px] justify-center", getStatusStyle(contract.status))}>
                  {getStatusLabel(contract.status)}
                </Badge>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// Risk Summary Cards
// ============================================================================

function RiskSummaryCards() {
  const risks = [
    {
      title: "Price Volatility Risk",
      level: "Medium",
      score: 45,
      description: "Energy commodities showing elevated volatility",
      icon: TrendingUp,
      color: "amber",
    },
    {
      title: "Supply Chain Risk",
      level: "Low",
      score: 28,
      description: "All major supply routes operational",
      icon: Truck,
      color: "green",
    },
    {
      title: "Geopolitical Risk",
      level: "Medium",
      score: 52,
      description: "Monitor Middle East developments",
      icon: Globe,
      color: "amber",
    },
    {
      title: "Counterparty Risk",
      level: "Low",
      score: 22,
      description: "All suppliers meeting obligations",
      icon: Users,
      color: "green",
    },
  ];

  const colorMap = {
    green: { bg: "bg-green-50", bar: "bg-green-500", text: "text-green-700", border: "border-green-200" },
    amber: { bg: "bg-amber-50", bar: "bg-amber-500", text: "text-amber-700", border: "border-amber-200" },
    red: { bg: "bg-red-50", bar: "bg-red-500", text: "text-red-700", border: "border-red-200" },
  };

  return (
    <div className="grid grid-cols-4 gap-4">
      {risks.map((risk) => {
        const colors = colorMap[risk.color as keyof typeof colorMap];
        return (
          <Card key={risk.title} className={cn("border shadow-sm", colors.border, colors.bg)}>
            <CardContent className="p-4">
              <div className="flex items-start justify-between mb-3">
                <risk.icon className={cn("w-5 h-5", colors.text)} />
                <Badge className={cn("text-[10px]", colors.bg, colors.text, colors.border)}>
                  {risk.level}
                </Badge>
              </div>
              <p className="font-medium text-slate-900 text-sm mb-1">{risk.title}</p>
              <div className="h-1.5 bg-white rounded-full overflow-hidden mb-2">
                <div
                  className={cn("h-full rounded-full", colors.bar)}
                  style={{ width: `${risk.score}%` }}
                />
              </div>
              <p className="text-xs text-slate-600">{risk.description}</p>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

// ============================================================================
// Main Dashboard
// ============================================================================

export function ProcurementDashboard() {
  const [activeTab, setActiveTab] = useState("overview");

  const forecasts = useMemo(() => generateForecasts(), []);
  const costBasis = useMemo(() => generateCostBasis(), []);
  const suppliers = useMemo(() => generateSupplierExposure(), []);
  const compliance = useMemo(() => generateComplianceItems(), []);
  const budget = useMemo(() => generateBudgetItems(), []);
  const contracts = useMemo(() => generateContractExpirations(), []);

  return (
    <LightModeWrapper>
      {/* Header */}
      <DashboardHeader />

      {/* Summary Cards */}
      <SummaryCards />

      {/* Risk Summary */}
      <div className="px-6 mb-6">
        <RiskSummaryCards />
      </div>

      {/* Main Content Tabs */}
      <div className="px-6 pb-6">
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="bg-white border border-slate-200 p-1 mb-6">
            <TabsTrigger
              value="overview"
              className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
            >
              Overview
            </TabsTrigger>
            <TabsTrigger
              value="forecasts"
              className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
            >
              Price Forecasts
            </TabsTrigger>
            <TabsTrigger
              value="suppliers"
              className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
            >
              Suppliers
            </TabsTrigger>
            <TabsTrigger
              value="compliance"
              className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
            >
              Compliance
            </TabsTrigger>
            <TabsTrigger
              value="budget"
              className="data-[state=active]:bg-blue-600 data-[state=active]:text-white"
            >
              Budget
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6 mt-0">
            <div className="grid grid-cols-12 gap-6">
              <div className="col-span-7">
                <PriceForecastPanel forecasts={forecasts} />
              </div>
              <div className="col-span-5">
                <ContractCalendar contracts={contracts} />
              </div>
            </div>
            <div className="grid grid-cols-12 gap-6">
              <div className="col-span-6">
                <CostBasisPanel data={costBasis} />
              </div>
              <div className="col-span-6">
                <ComplianceChecklist items={compliance} />
              </div>
            </div>
          </TabsContent>

          {/* Forecasts Tab */}
          <TabsContent value="forecasts" className="space-y-6 mt-0">
            <div className="grid grid-cols-12 gap-6">
              <div className="col-span-8">
                <PriceForecastPanel forecasts={forecasts} />
              </div>
              <div className="col-span-4">
                <CostBasisPanel data={costBasis} />
              </div>
            </div>
          </TabsContent>

          {/* Suppliers Tab */}
          <TabsContent value="suppliers" className="space-y-6 mt-0">
            <div className="grid grid-cols-12 gap-6">
              <div className="col-span-7">
                <SupplierExposureHeatmap data={suppliers} />
              </div>
              <div className="col-span-5">
                <ContractCalendar contracts={contracts} />
              </div>
            </div>
          </TabsContent>

          {/* Compliance Tab */}
          <TabsContent value="compliance" className="space-y-6 mt-0">
            <ComplianceChecklist items={compliance} />
          </TabsContent>

          {/* Budget Tab */}
          <TabsContent value="budget" className="space-y-6 mt-0">
            <div className="grid grid-cols-12 gap-6">
              <div className="col-span-7">
                <BudgetVsActualPanel data={budget} />
              </div>
              <div className="col-span-5">
                <CostBasisPanel data={costBasis} />
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>

      {/* Footer */}
      <div className="border-t border-slate-200 bg-white px-6 py-3">
        <div className="flex items-center justify-between text-xs text-slate-500">
          <span>QDT Nexus Procurement Module v2.1</span>
          <span>Data refreshes every 5 minutes</span>
        </div>
      </div>
    </LightModeWrapper>
  );
}

export default ProcurementDashboard;
