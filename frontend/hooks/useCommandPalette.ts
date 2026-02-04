"use client";

import { useState, useCallback, useMemo, useEffect } from "react";
import type { Command, CommandGroup } from "@/types/commands";
import { DEFAULT_COMMANDS, groupCommandsByCategory } from "@/types/commands";

interface UseCommandPaletteOptions {
  commands?: Command[];
  onExecute?: (command: Command) => void;
}

interface UseCommandPaletteReturn {
  isOpen: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  filteredCommands: Command[];
  groupedCommands: CommandGroup[];
  selectedIndex: number;
  setSelectedIndex: (index: number) => void;
  selectNext: () => void;
  selectPrevious: () => void;
  executeSelected: () => void;
  executeCommand: (command: Command) => void;
}

function fuzzyMatch(text: string, query: string): boolean {
  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase();

  // Direct substring match
  if (lowerText.includes(lowerQuery)) return true;

  // Fuzzy character matching
  let queryIndex = 0;
  for (let i = 0; i < lowerText.length && queryIndex < lowerQuery.length; i++) {
    if (lowerText[i] === lowerQuery[queryIndex]) {
      queryIndex++;
    }
  }
  return queryIndex === lowerQuery.length;
}

function scoreMatch(command: Command, query: string): number {
  if (!query) return 0;

  const lowerQuery = query.toLowerCase();
  let score = 0;

  // Label exact match (highest priority)
  if (command.label.toLowerCase() === lowerQuery) {
    score += 100;
  }
  // Label starts with query
  else if (command.label.toLowerCase().startsWith(lowerQuery)) {
    score += 80;
  }
  // Label contains query
  else if (command.label.toLowerCase().includes(lowerQuery)) {
    score += 60;
  }

  // Keyword matches
  for (const keyword of command.keywords) {
    if (keyword.toLowerCase() === lowerQuery) {
      score += 40;
    } else if (keyword.toLowerCase().startsWith(lowerQuery)) {
      score += 20;
    } else if (keyword.toLowerCase().includes(lowerQuery)) {
      score += 10;
    }
  }

  return score;
}

export function useCommandPalette(
  options: UseCommandPaletteOptions = {}
): UseCommandPaletteReturn {
  const { commands = DEFAULT_COMMANDS, onExecute } = options;

  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);

  const open = useCallback(() => {
    setIsOpen(true);
    setSearchQuery("");
    setSelectedIndex(0);
  }, []);

  const close = useCallback(() => {
    setIsOpen(false);
    setSearchQuery("");
    setSelectedIndex(0);
  }, []);

  const toggle = useCallback(() => {
    if (isOpen) {
      close();
    } else {
      open();
    }
  }, [isOpen, open, close]);

  const filteredCommands = useMemo(() => {
    if (!searchQuery) return commands;

    const matchedCommands = commands.filter((command) => {
      // Check label
      if (fuzzyMatch(command.label, searchQuery)) return true;

      // Check keywords
      for (const keyword of command.keywords) {
        if (fuzzyMatch(keyword, searchQuery)) return true;
      }

      return false;
    });

    // Sort by match score
    return matchedCommands.sort((a, b) => {
      const scoreA = scoreMatch(a, searchQuery);
      const scoreB = scoreMatch(b, searchQuery);
      return scoreB - scoreA;
    });
  }, [commands, searchQuery]);

  const groupedCommands = useMemo(() => {
    return groupCommandsByCategory(filteredCommands);
  }, [filteredCommands]);

  // Reset selected index when filtered commands change
  useEffect(() => {
    setSelectedIndex(0);
  }, [filteredCommands]);

  const selectNext = useCallback(() => {
    setSelectedIndex((prev) =>
      prev < filteredCommands.length - 1 ? prev + 1 : 0
    );
  }, [filteredCommands.length]);

  const selectPrevious = useCallback(() => {
    setSelectedIndex((prev) =>
      prev > 0 ? prev - 1 : filteredCommands.length - 1
    );
  }, [filteredCommands.length]);

  const executeCommand = useCallback(
    (command: Command) => {
      onExecute?.(command);
      close();
    },
    [onExecute, close]
  );

  const executeSelected = useCallback(() => {
    const selectedCommand = filteredCommands[selectedIndex];
    if (selectedCommand) {
      executeCommand(selectedCommand);
    }
  }, [filteredCommands, selectedIndex, executeCommand]);

  // Global keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd+K or Ctrl+K to open
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        toggle();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [toggle]);

  return {
    isOpen,
    open,
    close,
    toggle,
    searchQuery,
    setSearchQuery,
    filteredCommands,
    groupedCommands,
    selectedIndex,
    setSelectedIndex,
    selectNext,
    selectPrevious,
    executeSelected,
    executeCommand,
  };
}
