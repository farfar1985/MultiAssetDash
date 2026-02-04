"use client";

import { CommandPaletteProvider } from "@/contexts/CommandPaletteContext";
import { CommandPalette } from "./CommandPalette";

interface CommandPaletteWrapperProps {
  children: React.ReactNode;
}

export function CommandPaletteWrapper({ children }: CommandPaletteWrapperProps) {
  return (
    <CommandPaletteProvider>
      {children}
      <CommandPalette />
    </CommandPaletteProvider>
  );
}
