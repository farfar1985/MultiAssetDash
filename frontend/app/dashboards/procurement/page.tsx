import { ProcurementDashboard } from "@/components/dashboard/ProcurementDashboard";

export const metadata = {
  title: "Procurement Dashboard | QDT Nexus",
  description: "Enterprise commodity procurement dashboard with price forecasts, supplier exposure, compliance tracking, and budget management",
};

export default function ProcurementPage() {
  return <ProcurementDashboard />;
}
