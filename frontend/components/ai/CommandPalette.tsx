"use client";

import { useEffect, useRef, useCallback } from "react";
import { createPortal } from "react-dom";
import { Search, Command as CommandIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { useCommandPalette } from "@/hooks/useCommandPalette";
import { useCommandPaletteContext } from "@/contexts/CommandPaletteContext";
import { CommandItem, CommandGroupHeader } from "./CommandItem";
import type { Command } from "@/types/commands";

export function CommandPalette() {
  const { commands, executeCommand } = useCommandPaletteContext();
  const {
    isOpen,
    close,
    searchQuery,
    setSearchQuery,
    filteredCommands,
    groupedCommands,
    selectedIndex,
    setSelectedIndex,
    selectNext,
    selectPrevious,
    executeSelected,
  } = useCommandPalette({
    commands,
    onExecute: executeCommand,
  });

  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const backdropRef = useRef<HTMLDivElement>(null);

  // Focus input when opening
  useEffect(() => {
    if (isOpen) {
      // Small delay to ensure the element is rendered
      requestAnimationFrame(() => {
        inputRef.current?.focus();
      });
    }
  }, [isOpen]);

  // Scroll selected item into view
  useEffect(() => {
    if (!listRef.current) return;

    const selectedElement = listRef.current.querySelector(
      '[aria-selected="true"]'
    );
    if (selectedElement) {
      selectedElement.scrollIntoView({ block: "nearest" });
    }
  }, [selectedIndex]);

  // Keyboard navigation
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          selectNext();
          break;
        case "ArrowUp":
          e.preventDefault();
          selectPrevious();
          break;
        case "Enter":
          e.preventDefault();
          executeSelected();
          break;
        case "Escape":
          e.preventDefault();
          close();
          break;
        case "Tab":
          e.preventDefault();
          if (e.shiftKey) {
            selectPrevious();
          } else {
            selectNext();
          }
          break;
      }
    },
    [selectNext, selectPrevious, executeSelected, close]
  );

  // Handle backdrop click
  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === backdropRef.current) {
        close();
      }
    },
    [close]
  );

  // Calculate flat index for each command
  const getFlatIndex = (groupIndex: number, itemIndex: number): number => {
    let index = 0;
    for (let i = 0; i < groupIndex; i++) {
      index += groupedCommands[i].commands.length;
    }
    return index + itemIndex;
  };

  const handleSelect = useCallback(
    (command: Command) => {
      executeCommand(command);
      close();
    },
    [executeCommand, close]
  );

  if (!isOpen) return null;

  // Use portal to render at document root
  return createPortal(
    <div
      ref={backdropRef}
      className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh] bg-black/60 backdrop-blur-sm animate-in fade-in duration-150"
      onClick={handleBackdropClick}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
      data-testid="command-palette"
    >
      <div
        className={cn(
          "w-full max-w-xl bg-neutral-900 rounded-xl shadow-2xl border border-neutral-800",
          "animate-in fade-in slide-in-from-top-4 duration-200",
          "overflow-hidden"
        )}
      >
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-neutral-800">
          <Search className="h-5 w-5 text-neutral-500 shrink-0" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Type a command or search..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            className={cn(
              "flex-1 bg-transparent text-neutral-100 placeholder-neutral-500",
              "text-sm outline-none"
            )}
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck={false}
          />
          <div className="flex items-center gap-1">
            <kbd className="hidden sm:inline-flex h-5 items-center gap-1 rounded border border-neutral-700 bg-neutral-800 px-1.5 font-mono text-[10px] font-medium text-neutral-500">
              esc
            </kbd>
          </div>
        </div>

        {/* Command list */}
        <div
          ref={listRef}
          className="max-h-80 overflow-y-auto py-2 scrollbar-thin scrollbar-thumb-neutral-700 scrollbar-track-transparent"
          role="listbox"
        >
          {filteredCommands.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm text-neutral-500">
              No commands found
            </div>
          ) : (
            groupedCommands.map((group, groupIndex) => (
              <div key={group.category} className="mb-2">
                <CommandGroupHeader
                  label={group.label}
                  category={group.category}
                />
                {group.commands.map((command, itemIndex) => {
                  const flatIndex = getFlatIndex(groupIndex, itemIndex);
                  return (
                    <CommandItem
                      key={command.id}
                      command={command}
                      isSelected={flatIndex === selectedIndex}
                      searchQuery={searchQuery}
                      onSelect={handleSelect}
                      onHover={() => setSelectedIndex(flatIndex)}
                    />
                  );
                })}
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-neutral-800 bg-neutral-900/80">
          <div className="flex items-center gap-4 text-xs text-neutral-500">
            <span className="flex items-center gap-1">
              <kbd className="inline-flex h-4 items-center rounded border border-neutral-700 bg-neutral-800 px-1 font-mono text-[10px]">
                ↑
              </kbd>
              <kbd className="inline-flex h-4 items-center rounded border border-neutral-700 bg-neutral-800 px-1 font-mono text-[10px]">
                ↓
              </kbd>
              <span>navigate</span>
            </span>
            <span className="flex items-center gap-1">
              <kbd className="inline-flex h-4 items-center rounded border border-neutral-700 bg-neutral-800 px-1 font-mono text-[10px]">
                ↵
              </kbd>
              <span>select</span>
            </span>
          </div>
          <div className="flex items-center gap-1 text-xs text-neutral-500">
            <CommandIcon className="h-3 w-3" />
            <span>K to open</span>
          </div>
        </div>
      </div>
    </div>,
    document.body
  );
}
