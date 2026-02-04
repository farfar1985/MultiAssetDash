"use client";

import { usePathname } from "next/navigation";
import { PERSONAS, ASSETS, type PersonaId, type AssetId } from "@/types";
import { EnsembleSelector } from "@/components/dashboard/EnsembleSelector";
import { useEnsembleMethod } from "@/contexts/EnsembleContext";

export function Header() {
  const pathname = usePathname();
  const [ensembleMethod, setEnsembleMethod] = useEnsembleMethod();

  // Parse current context from pathname
  const pathSegments = pathname.split("/").filter(Boolean);
  const personaId = pathSegments[1] as PersonaId | undefined;
  const assetId = pathSegments[2] as AssetId | undefined;

  const persona = personaId ? PERSONAS[personaId] : null;
  const asset = assetId ? ASSETS[assetId] : null;

  // Build page title
  let pageTitle = "Dashboard";
  if (asset) {
    pageTitle = asset.name;
  } else if (persona) {
    pageTitle = persona.name;
  }

  return (
    <header className="h-14 border-b border-neutral-800 bg-neutral-950 flex items-center justify-between px-6">
      <div className="flex items-center gap-4">
        <h1 className="text-lg font-semibold text-neutral-100">
          {pageTitle}
        </h1>
        {persona && !asset && (
          <span className="text-sm text-neutral-500">
            {persona.description}
          </span>
        )}
      </div>

      <div className="flex items-center gap-4">
        {/* Ensemble Method Selector */}
        <div className="flex items-center gap-2">
          <span className="text-xs text-neutral-500">Ensemble:</span>
          <EnsembleSelector
            value={ensembleMethod}
            onChange={setEnsembleMethod}
            variant="dropdown"
          />
        </div>

        {/* Status Indicator */}
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span className="text-xs text-neutral-500">Live</span>
        </div>

        {/* Quick Actions - Placeholder */}
        <button className="px-3 py-1.5 text-sm bg-neutral-800 hover:bg-neutral-700 rounded-md text-neutral-300 transition-colors">
          Settings
        </button>
      </div>
    </header>
  );
}
