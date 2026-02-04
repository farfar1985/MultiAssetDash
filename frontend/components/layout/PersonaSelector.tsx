"use client";

import { useState } from "react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { PERSONAS, type PersonaId } from "@/types";

interface PersonaSelectorProps {
  currentPersona: string;
}

export function PersonaSelector({ currentPersona }: PersonaSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const current = PERSONAS[currentPersona as PersonaId];

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3 py-2 bg-neutral-900 border border-neutral-800 rounded-md text-sm text-neutral-200 hover:border-neutral-700 transition-colors"
      >
        <span>{current?.name || "Select Persona"}</span>
        <svg
          className={cn(
            "w-4 h-4 text-neutral-500 transition-transform",
            isOpen && "rotate-180"
          )}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 9l-7 7-7-7"
          />
        </svg>
      </button>

      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />

          {/* Dropdown */}
          <div className="absolute top-full left-0 right-0 mt-1 z-20 bg-neutral-900 border border-neutral-800 rounded-md shadow-lg py-1 max-h-80 overflow-y-auto">
            {Object.values(PERSONAS).map((persona) => (
              <Link
                key={persona.id}
                href={`/dashboard/${persona.id}`}
                onClick={() => setIsOpen(false)}
                className={cn(
                  "block px-3 py-2 text-sm transition-colors",
                  persona.id === currentPersona
                    ? "bg-blue-500/10 text-blue-500"
                    : "text-neutral-300 hover:bg-neutral-800"
                )}
              >
                <div className="font-medium">{persona.name}</div>
                <div className="text-xs text-neutral-500 mt-0.5">
                  {persona.description}
                </div>
              </Link>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
