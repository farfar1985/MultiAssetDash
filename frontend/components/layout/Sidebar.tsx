"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { ASSETS, type AssetId } from "@/types";
import { PersonaSelector } from "./PersonaSelector";

const assetGroups = {
  energy: ["crude-oil", "natural-gas"] as AssetId[],
  metals: ["gold", "silver", "copper", "platinum"] as AssetId[],
  crypto: ["bitcoin"] as AssetId[],
  agriculture: ["wheat", "corn", "soybean"] as AssetId[],
};

export function Sidebar() {
  const pathname = usePathname();

  // Extract current persona from path
  const pathSegments = pathname.split("/");
  const currentPersona = pathSegments[2] || "quant";

  return (
    <aside className="w-64 bg-sidebar-bg border-r border-neutral-800 flex flex-col h-full">
      {/* Logo */}
      <div className="p-4 border-b border-neutral-800">
        <Link href="/dashboard" className="flex items-center gap-2">
          <div className="w-8 h-8 bg-blue-500 rounded flex items-center justify-center">
            <span className="font-bold text-white text-sm">QN</span>
          </div>
          <span className="font-semibold text-lg text-neutral-100">
            QDT Nexus
          </span>
        </Link>
      </div>

      {/* Persona Selector */}
      <div className="p-4 border-b border-neutral-800">
        <PersonaSelector currentPersona={currentPersona} />
      </div>

      {/* Asset Navigation */}
      <nav className="flex-1 overflow-y-auto p-4">
        <div className="space-y-6">
          {Object.entries(assetGroups).map(([category, assets]) => (
            <div key={category}>
              <h3 className="text-xs font-semibold uppercase tracking-wider text-neutral-500 mb-2">
                {category}
              </h3>
              <ul className="space-y-1">
                {assets.map((assetId) => {
                  const asset = ASSETS[assetId];
                  const href = `/dashboard/${currentPersona}/${assetId}`;
                  const isActive = pathname === href;

                  return (
                    <li key={assetId}>
                      <Link
                        href={href}
                        className={cn(
                          "flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors",
                          isActive
                            ? "bg-blue-500/10 text-blue-500"
                            : "text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/50"
                        )}
                      >
                        <span className="font-mono text-xs w-8 text-neutral-500">
                          {asset.symbol}
                        </span>
                        <span>{asset.name}</span>
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </div>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-neutral-800">
        <div className="text-xs text-neutral-600">
          QDT Nexus v0.1.0
        </div>
      </div>
    </aside>
  );
}
