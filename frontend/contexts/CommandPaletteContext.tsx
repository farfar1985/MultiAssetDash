"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useMemo,
  type ReactNode,
} from "react";
import { useRouter } from "next/navigation";
import type { Command, CommandAction, FilterConfig } from "@/types/commands";
import { DEFAULT_COMMANDS } from "@/types/commands";
import { useEnsembleContext } from "./EnsembleContext";
import type { EnsembleMethod } from "@/lib/api-client";

interface CommandPaletteContextValue {
  commands: Command[];
  registerCommand: (command: Command) => void;
  unregisterCommand: (commandId: string) => void;
  executeCommand: (command: Command) => void;
  activeFilters: FilterConfig;
  setActiveFilters: (filters: FilterConfig) => void;
  clearFilters: () => void;
}

const CommandPaletteContext = createContext<CommandPaletteContextValue | null>(
  null
);

interface CommandPaletteProviderProps {
  children: ReactNode;
}

export function CommandPaletteProvider({
  children,
}: CommandPaletteProviderProps) {
  const router = useRouter();
  const { setMethod } = useEnsembleContext();
  const [customCommands, setCustomCommands] = useState<Command[]>([]);
  const [activeFilters, setActiveFilters] = useState<FilterConfig>({});

  const commands = useMemo(() => {
    return [...DEFAULT_COMMANDS, ...customCommands];
  }, [customCommands]);

  const registerCommand = useCallback((command: Command) => {
    setCustomCommands((prev) => {
      // Avoid duplicates
      if (prev.some((c) => c.id === command.id)) {
        return prev.map((c) => (c.id === command.id ? command : c));
      }
      return [...prev, command];
    });
  }, []);

  const unregisterCommand = useCallback((commandId: string) => {
    setCustomCommands((prev) => prev.filter((c) => c.id !== commandId));
  }, []);

  const executeAction = useCallback(
    (action: CommandAction) => {
      switch (action.type) {
        case "navigate":
          router.push(action.path);
          break;

        case "navigate-asset":
          router.push(`/dashboard/asset/${action.assetId}`);
          break;

        case "action": {
          const handler = action.handler;

          // Handle ensemble method changes
          if (handler.startsWith("setEnsemble:")) {
            const method = handler.split(":")[1] as EnsembleMethod;
            setMethod(method);
            break;
          }

          // Handle other actions
          switch (handler) {
            case "refreshSignals":
              // Dispatch a custom event that components can listen to
              window.dispatchEvent(new CustomEvent("command:refreshSignals"));
              break;
            case "exportData":
              window.dispatchEvent(new CustomEvent("command:exportData"));
              break;
            default:
              console.warn(`Unknown action handler: ${handler}`);
          }
          break;
        }

        case "ai-query":
          // Dispatch AI query event for AI assistant to handle
          window.dispatchEvent(
            new CustomEvent("command:aiQuery", {
              detail: { query: action.query },
            })
          );
          break;

        case "filter":
          if (Object.keys(action.filter).length === 0) {
            setActiveFilters({});
          } else {
            setActiveFilters((prev) => ({ ...prev, ...action.filter }));
          }
          // Dispatch filter change event
          window.dispatchEvent(
            new CustomEvent("command:filterChange", {
              detail: { filters: action.filter },
            })
          );
          break;
      }
    },
    [router, setMethod]
  );

  const executeCommand = useCallback(
    (command: Command) => {
      executeAction(command.action);
    },
    [executeAction]
  );

  const clearFilters = useCallback(() => {
    setActiveFilters({});
    window.dispatchEvent(
      new CustomEvent("command:filterChange", {
        detail: { filters: {} },
      })
    );
  }, []);

  const value = useMemo<CommandPaletteContextValue>(
    () => ({
      commands,
      registerCommand,
      unregisterCommand,
      executeCommand,
      activeFilters,
      setActiveFilters,
      clearFilters,
    }),
    [
      commands,
      registerCommand,
      unregisterCommand,
      executeCommand,
      activeFilters,
      clearFilters,
    ]
  );

  return (
    <CommandPaletteContext.Provider value={value}>
      {children}
    </CommandPaletteContext.Provider>
  );
}

export function useCommandPaletteContext(): CommandPaletteContextValue {
  const context = useContext(CommandPaletteContext);
  if (!context) {
    throw new Error(
      "useCommandPaletteContext must be used within a CommandPaletteProvider"
    );
  }
  return context;
}
