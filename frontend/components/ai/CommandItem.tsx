"use client";

import { memo } from "react";
import {
  LayoutDashboard,
  Droplet,
  Bitcoin,
  Gem,
  Flame,
  Circle,
  Settings,
  RefreshCw,
  Download,
  Target,
  TrendingUp,
  TrendingDown,
  Layers,
  MessageCircle,
  HelpCircle,
  GitCompare,
  FileText,
  AlertTriangle,
  Star,
  Zap,
  Filter,
  X,
  Compass,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { Command, CommandCategory } from "@/types/commands";

const ICON_MAP: Record<string, LucideIcon> = {
  "layout-dashboard": LayoutDashboard,
  droplet: Droplet,
  bitcoin: Bitcoin,
  gem: Gem,
  flame: Flame,
  circle: Circle,
  settings: Settings,
  "refresh-cw": RefreshCw,
  download: Download,
  target: Target,
  "trending-up": TrendingUp,
  "trending-down": TrendingDown,
  layers: Layers,
  "message-circle": MessageCircle,
  "help-circle": HelpCircle,
  "git-compare": GitCompare,
  "file-text": FileText,
  "alert-triangle": AlertTriangle,
  star: Star,
  zap: Zap,
  filter: Filter,
  x: X,
  compass: Compass,
  sparkles: Sparkles,
};

const CATEGORY_COLORS: Record<CommandCategory, string> = {
  navigation: "text-blue-400",
  action: "text-amber-400",
  "ai-query": "text-purple-400",
  filter: "text-emerald-400",
};

interface CommandItemProps {
  command: Command;
  isSelected: boolean;
  searchQuery: string;
  onSelect: (command: Command) => void;
  onHover?: () => void;
}

function highlightMatch(text: string, query: string): React.ReactNode {
  if (!query) return text;

  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase();
  const index = lowerText.indexOf(lowerQuery);

  if (index === -1) return text;

  return (
    <>
      {text.slice(0, index)}
      <span className="bg-amber-500/30 text-amber-300">
        {text.slice(index, index + query.length)}
      </span>
      {text.slice(index + query.length)}
    </>
  );
}

export const CommandItem = memo(function CommandItem({
  command,
  isSelected,
  searchQuery,
  onSelect,
  onHover,
}: CommandItemProps) {
  const Icon = command.icon ? ICON_MAP[command.icon] : null;

  const handleClick = () => {
    onSelect(command);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect(command);
    }
  };

  const handleMouseEnter = () => {
    onHover?.();
  };

  return (
    <div
      role="option"
      aria-selected={isSelected}
      tabIndex={isSelected ? 0 : -1}
      data-testid="command-item"
      data-highlighted={isSelected ? "true" : "false"}
      className={cn(
        "flex items-center gap-3 px-3 py-2.5 cursor-pointer rounded-md mx-1",
        "transition-colors duration-75",
        isSelected
          ? "bg-neutral-800 text-neutral-100"
          : "text-neutral-400 hover:bg-neutral-800/50 hover:text-neutral-200"
      )}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      onMouseEnter={handleMouseEnter}
    >
      {Icon && (
        <Icon
          className={cn("h-4 w-4 shrink-0", CATEGORY_COLORS[command.category])}
        />
      )}
      <span className="flex-1 truncate text-sm">
        {highlightMatch(command.label, searchQuery)}
      </span>
      {command.shortcut && (
        <kbd className="hidden sm:inline-flex h-5 items-center gap-1 rounded border border-neutral-700 bg-neutral-800 px-1.5 font-mono text-[10px] font-medium text-neutral-500">
          {command.shortcut}
        </kbd>
      )}
    </div>
  );
});

interface CommandGroupHeaderProps {
  label: string;
  category: CommandCategory;
}

export const CommandGroupHeader = memo(function CommandGroupHeader({
  label,
  category,
}: CommandGroupHeaderProps) {
  const Icon = ICON_MAP[getCategoryIcon(category)];

  return (
    <div className="flex items-center gap-2 px-3 py-2 text-xs font-medium text-neutral-500 uppercase tracking-wider">
      {Icon && <Icon className={cn("h-3 w-3", CATEGORY_COLORS[category])} />}
      {label}
    </div>
  );
});

function getCategoryIcon(category: CommandCategory): string {
  switch (category) {
    case "navigation":
      return "compass";
    case "action":
      return "zap";
    case "ai-query":
      return "sparkles";
    case "filter":
      return "filter";
    default:
      return "circle";
  }
}
