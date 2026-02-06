"use client";

import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  Shield,
  Briefcase,
  Brain,
  Truck,
  Zap,
  GraduationCap,
  Sparkles,
  ChevronRight,
  Users,
  Target,
  Activity,
  Globe,
  BarChart3,
} from "lucide-react";

interface DashboardCard {
  id: string;
  name: string;
  description: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  bgColor: string;
  borderColor: string;
  audience: string;
  features: string[];
}

const DASHBOARDS: DashboardCard[] = [
  {
    id: "retail",
    name: "Retail Dashboard",
    description: "Mobile-first, beginner-friendly signals in plain English",
    href: "/dashboards/retail",
    icon: Sparkles,
    color: "text-orange-400",
    bgColor: "bg-orange-500/10",
    borderColor: "border-orange-500/30 hover:border-orange-500/50",
    audience: "Beginners",
    features: ["Plain English", "Mobile-first", "Swipeable cards"],
  },
  {
    id: "pro-retail",
    name: "Pro Retail",
    description: "Educational dashboard - learn while you trade with signal explanations",
    href: "/dashboards/pro-retail",
    icon: GraduationCap,
    color: "text-cyan-400",
    bgColor: "bg-cyan-500/10",
    borderColor: "border-cyan-500/30 hover:border-cyan-500/50",
    audience: "Learning Traders",
    features: ["Why This Signal", "Glossary", "Learning Progress"],
  },
  {
    id: "alpha-gen-pro",
    name: "Alpha Gen Pro",
    description: "Professional-grade ensemble signals for alpha generation",
    href: "/dashboards/alpha-gen-pro",
    icon: Zap,
    color: "text-purple-400",
    bgColor: "bg-purple-500/10",
    borderColor: "border-purple-500/30 hover:border-purple-500/50",
    audience: "Active Traders",
    features: ["Live API Data", "Multi-asset", "Position Sizing"],
  },
  {
    id: "hedge-fund",
    name: "Hedge Fund Portfolio",
    description: "Multi-asset institutional dashboard with performance attribution",
    href: "/dashboards/hedge-fund",
    icon: Briefcase,
    color: "text-indigo-400",
    bgColor: "bg-indigo-500/10",
    borderColor: "border-indigo-500/30 hover:border-indigo-500/50",
    audience: "Portfolio Managers",
    features: ["AUM Tracking", "Factor Attribution", "Benchmark Comparison"],
  },
  {
    id: "hedging",
    name: "Hedging Dashboard",
    description: "Risk management with hedge ratios, Greeks, and correlation analysis",
    href: "/dashboards/hedging",
    icon: Shield,
    color: "text-emerald-400",
    bgColor: "bg-emerald-500/10",
    borderColor: "border-emerald-500/30 hover:border-emerald-500/50",
    audience: "Risk Managers",
    features: ["Hedge Ratios", "Options Greeks", "Factor Attribution"],
  },
  {
    id: "procurement",
    name: "Procurement",
    description: "Commodity procurement dashboard for supply chain professionals",
    href: "/dashboards/procurement",
    icon: Truck,
    color: "text-amber-400",
    bgColor: "bg-amber-500/10",
    borderColor: "border-amber-500/30 hover:border-amber-500/50",
    audience: "Procurement Teams",
    features: ["Forward Curves", "Budget Impact", "Contract Timing"],
  },
  {
    id: "hardcore-quant",
    name: "Hardcore Quant",
    description: "Deep quantitative analysis with regime detection and entropy metrics",
    href: "/dashboards/hardcore-quant",
    icon: Brain,
    color: "text-pink-400",
    bgColor: "bg-pink-500/10",
    borderColor: "border-pink-500/30 hover:border-pink-500/50",
    audience: "Quant Researchers",
    features: ["Regime Analysis", "Entropy Metrics", "Signal Decomposition"],
  },
  {
    id: "regimes",
    name: "Market Regimes",
    description: "Multi-asset HMM regime monitor across commodities, crypto, and indices",
    href: "/dashboards/regimes",
    icon: Globe,
    color: "text-blue-400",
    bgColor: "bg-blue-500/10",
    borderColor: "border-blue-500/30 hover:border-blue-500/50",
    audience: "All Users",
    features: ["13 Assets", "HMM Detection", "Category Filters"],
  },
  {
    id: "backtest",
    name: "Walk-Forward Backtest",
    description: "Compare ensemble methods with walk-forward validation and cost analysis",
    href: "/dashboards/backtest",
    icon: BarChart3,
    color: "text-green-400",
    bgColor: "bg-green-500/10",
    borderColor: "border-green-500/30 hover:border-green-500/50",
    audience: "Quant Analysts",
    features: ["Walk-Forward", "Regime Analysis", "Cost Modeling"],
  },
];

export default function DashboardsIndexPage() {
  return (
    <div className="min-h-screen bg-neutral-950 p-6 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-neutral-100">Persona Dashboards</h1>
          <p className="text-neutral-400 mt-1">
            Choose the dashboard that matches your trading style and experience
          </p>
        </div>
        <Badge className="bg-green-500/20 text-green-400 border-green-500/30 px-4 py-2">
          <Activity className="w-4 h-4 mr-2 animate-pulse" />
          All Systems Live
        </Badge>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-3 gap-4">
        <Card className="bg-neutral-900/50 border-neutral-800">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="p-3 bg-blue-500/10 rounded-xl">
              <Users className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <div className="text-2xl font-bold text-neutral-100">9</div>
              <div className="text-sm text-neutral-500">Persona Dashboards</div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-neutral-900/50 border-neutral-800">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="p-3 bg-purple-500/10 rounded-xl">
              <Brain className="w-6 h-6 text-purple-400" />
            </div>
            <div>
              <div className="text-2xl font-bold text-neutral-100">10,179</div>
              <div className="text-sm text-neutral-500">AI Models</div>
            </div>
          </CardContent>
        </Card>
        <Card className="bg-neutral-900/50 border-neutral-800">
          <CardContent className="p-4 flex items-center gap-4">
            <div className="p-3 bg-green-500/10 rounded-xl">
              <Target className="w-6 h-6 text-green-400" />
            </div>
            <div>
              <div className="text-2xl font-bold text-neutral-100">13</div>
              <div className="text-sm text-neutral-500">Asset Classes</div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Dashboard Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {DASHBOARDS.map((dashboard) => {
          const Icon = dashboard.icon;
          return (
            <Link key={dashboard.id} href={dashboard.href}>
              <Card
                className={cn(
                  "border-2 transition-all duration-300 cursor-pointer group h-full",
                  "bg-neutral-900/50 hover:bg-neutral-900/80",
                  dashboard.borderColor
                )}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className={cn("p-3 rounded-xl", dashboard.bgColor)}>
                      <Icon className={cn("w-6 h-6", dashboard.color)} />
                    </div>
                    <Badge className="bg-neutral-800 text-neutral-400 border-neutral-700 text-xs">
                      {dashboard.audience}
                    </Badge>
                  </div>
                  <CardTitle className="text-lg font-semibold text-neutral-100 mt-3">
                    {dashboard.name}
                  </CardTitle>
                </CardHeader>
                <CardContent className="pt-0">
                  <p className="text-sm text-neutral-400 mb-4">{dashboard.description}</p>

                  {/* Features */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    {dashboard.features.map((feature) => (
                      <Badge
                        key={feature}
                        className={cn(
                          "text-xs",
                          dashboard.bgColor,
                          dashboard.color,
                          "border border-current/20"
                        )}
                      >
                        {feature}
                      </Badge>
                    ))}
                  </div>

                  {/* CTA */}
                  <div className="flex items-center text-sm font-medium text-neutral-400 group-hover:text-neutral-100 transition-colors">
                    Open Dashboard
                    <ChevronRight className="w-4 h-4 ml-1 group-hover:translate-x-1 transition-transform" />
                  </div>
                </CardContent>
              </Card>
            </Link>
          );
        })}
      </div>

      {/* Footer */}
      <div className="text-center text-sm text-neutral-600 pt-8 border-t border-neutral-800">
        QDT Nexus • Quantum-Driven Trading Platform • All dashboards powered by live ensemble signals
      </div>
    </div>
  );
}
