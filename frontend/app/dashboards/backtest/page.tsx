import type { Metadata } from "next";
import { BacktestDashboard } from "@/components/dashboard/BacktestDashboard";

export const metadata: Metadata = {
  title: "Walk-Forward Backtest | QDT Nexus",
  description:
    "Analyze ensemble method performance through walk-forward validation. Compare accuracy, Sharpe ratios, and cost-adjusted returns across market regimes.",
};

export default function BacktestPage() {
  return <BacktestDashboard />;
}
