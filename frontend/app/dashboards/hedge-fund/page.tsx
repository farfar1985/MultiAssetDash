import { HedgeFundDashboard } from "@/components/dashboard/HedgeFundDashboard";

export const metadata = {
  title: "Hedge Fund Portfolio | QDT Nexus",
  description: "Institutional multi-asset portfolio dashboard with performance attribution, benchmark comparison, and factor exposure analysis",
};

export default function HedgeFundPage() {
  return <HedgeFundDashboard />;
}
