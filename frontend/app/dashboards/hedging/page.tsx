import { HedgingDashboard } from "@/components/dashboard/HedgingDashboard";

export const metadata = {
  title: "Hedging Dashboard | QDT Nexus",
  description: "Institutional hedging dashboard with hedge ratios, options Greeks, factor attribution, and cross-asset correlation analysis for risk management.",
};

export default function HedgingPage() {
  return <HedgingDashboard />;
}
