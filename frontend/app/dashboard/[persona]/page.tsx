import { notFound } from "next/navigation";
import { PERSONAS, type PersonaId, ASSETS } from "@/types";
import { SignalCard } from "@/components/dashboard/SignalCard";
import { MetricsGrid } from "@/components/dashboard/MetricsGrid";
import { Separator } from "@/components/ui/separator";
import { AlphaProDashboard } from "@/components/dashboard/AlphaProDashboard";
import { HedgingDashboard } from "@/components/dashboard/HedgingDashboard";
import { QuantDashboard } from "@/components/dashboard/QuantDashboard";
import { RetailDashboard } from "@/components/dashboard/RetailDashboard";

interface PersonaDashboardProps {
  params: {
    persona: string;
  };
}

export function generateStaticParams() {
  return Object.keys(PERSONAS).map((persona) => ({
    persona,
  }));
}

export default function PersonaDashboardPage({ params }: PersonaDashboardProps) {
  const personaId = params.persona as PersonaId;
  const persona = PERSONAS[personaId];

  if (!persona) {
    notFound();
  }

  // Specialized dashboard for Alpha Gen Pro persona
  if (personaId === "alphapro") {
    return <AlphaProDashboard />;
  }

  // Specialized dashboard for Hedging Team persona
  if (personaId === "hedging") {
    return <HedgingDashboard />;
  }

  // Specialized dashboard for Quant persona
  if (personaId === "quant") {
    return <QuantDashboard />;
  }

  // Specialized dashboard for Retail persona
  if (personaId === "retail") {
    return <RetailDashboard />;
  }

  const allAssetIds = Object.keys(ASSETS) as (keyof typeof ASSETS)[];

  return (
    <div className="space-y-6">
      {/* Persona Header */}
      <div className="pb-4">
        <h1 className="text-2xl font-bold text-neutral-100">
          {persona.name}
        </h1>
        <p className="text-sm text-neutral-400 mt-1">
          {persona.description}
        </p>
      </div>

      <Separator className="bg-neutral-800" />

      {/* Metrics Overview */}
      <section>
        <h2 className="text-lg font-semibold text-neutral-200 mb-4">
          Performance Overview
        </h2>
        <MetricsGrid assetId="crude-oil" />
      </section>

      <Separator className="bg-neutral-800" />

      {/* Active Signals - All Assets */}
      <section>
        <h2 className="text-lg font-semibold text-neutral-200 mb-4">
          Active Signals
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {allAssetIds.map((assetId) => (
            <SignalCard
              key={assetId}
              assetId={assetId}
              defaultHorizon="D+1"
            />
          ))}
        </div>
      </section>
    </div>
  );
}
