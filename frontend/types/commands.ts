import type { AssetId } from "./index";
import type { EnsembleMethod } from "@/lib/api-client";

export type CommandCategory =
  | "navigation"
  | "action"
  | "ai-query"
  | "filter";

export interface Command {
  id: string;
  label: string;
  category: CommandCategory;
  keywords: string[];
  icon?: string;
  shortcut?: string;
  action: CommandAction;
}

export type CommandAction =
  | { type: "navigate"; path: string }
  | { type: "navigate-asset"; assetId: AssetId }
  | { type: "action"; handler: string }
  | { type: "ai-query"; query: string }
  | { type: "filter"; filter: FilterConfig };

export interface FilterConfig {
  direction?: "bullish" | "bearish" | "neutral";
  conviction?: "high" | "medium" | "low";
  category?: "energy" | "metals" | "crypto" | "agriculture";
}

export interface CommandGroup {
  category: CommandCategory;
  label: string;
  commands: Command[];
}

export const COMMAND_CATEGORIES: Record<CommandCategory, { label: string; icon: string }> = {
  navigation: { label: "Navigation", icon: "compass" },
  action: { label: "Actions", icon: "zap" },
  "ai-query": { label: "AI Queries", icon: "sparkles" },
  filter: { label: "Filters", icon: "filter" },
};

export const DEFAULT_COMMANDS: Command[] = [
  // Navigation commands
  {
    id: "nav-dashboard",
    label: "Go to Dashboard",
    category: "navigation",
    keywords: ["home", "main", "overview"],
    icon: "layout-dashboard",
    shortcut: "G D",
    action: { type: "navigate", path: "/dashboard" },
  },
  {
    id: "nav-crude-oil",
    label: "Go to Crude Oil",
    category: "navigation",
    keywords: ["oil", "cl", "energy", "wti"],
    icon: "droplet",
    action: { type: "navigate-asset", assetId: "crude-oil" },
  },
  {
    id: "nav-bitcoin",
    label: "Go to Bitcoin",
    category: "navigation",
    keywords: ["btc", "crypto", "cryptocurrency"],
    icon: "bitcoin",
    action: { type: "navigate-asset", assetId: "bitcoin" },
  },
  {
    id: "nav-gold",
    label: "Go to Gold",
    category: "navigation",
    keywords: ["gc", "metals", "precious"],
    icon: "gem",
    action: { type: "navigate-asset", assetId: "gold" },
  },
  {
    id: "nav-silver",
    label: "Go to Silver",
    category: "navigation",
    keywords: ["si", "metals", "precious"],
    icon: "gem",
    action: { type: "navigate-asset", assetId: "silver" },
  },
  {
    id: "nav-natural-gas",
    label: "Go to Natural Gas",
    category: "navigation",
    keywords: ["ng", "energy", "gas"],
    icon: "flame",
    action: { type: "navigate-asset", assetId: "natural-gas" },
  },
  {
    id: "nav-copper",
    label: "Go to Copper",
    category: "navigation",
    keywords: ["hg", "metals", "industrial"],
    icon: "circle",
    action: { type: "navigate-asset", assetId: "copper" },
  },
  {
    id: "nav-settings",
    label: "Open Settings",
    category: "navigation",
    keywords: ["preferences", "config", "options"],
    icon: "settings",
    shortcut: "G S",
    action: { type: "navigate", path: "/settings" },
  },

  // Action commands
  {
    id: "action-refresh",
    label: "Refresh Signals",
    category: "action",
    keywords: ["reload", "update", "fetch"],
    icon: "refresh-cw",
    shortcut: "R",
    action: { type: "action", handler: "refreshSignals" },
  },
  {
    id: "action-export",
    label: "Export Data",
    category: "action",
    keywords: ["download", "csv", "save"],
    icon: "download",
    shortcut: "E",
    action: { type: "action", handler: "exportData" },
  },
  {
    id: "action-ensemble-accuracy",
    label: "Change to Accuracy Weighted",
    category: "action",
    keywords: ["ensemble", "method", "weighted"],
    icon: "target",
    action: { type: "action", handler: "setEnsemble:accuracy_weighted" },
  },
  {
    id: "action-ensemble-sharpe",
    label: "Change to Top K Sharpe",
    category: "action",
    keywords: ["ensemble", "method", "sharpe", "risk"],
    icon: "trending-up",
    action: { type: "action", handler: "setEnsemble:top_k_sharpe" },
  },
  {
    id: "action-ensemble-hybrid",
    label: "Change to Hybrid Ensemble",
    category: "action",
    keywords: ["ensemble", "method", "combined"],
    icon: "layers",
    action: { type: "action", handler: "setEnsemble:hybrid_blend" },
  },

  // AI Query commands
  {
    id: "ai-explain-signal",
    label: "Explain Current Signal",
    category: "ai-query",
    keywords: ["why", "reason", "analysis"],
    icon: "message-circle",
    action: { type: "ai-query", query: "explain_signal" },
  },
  {
    id: "ai-crude-bullish",
    label: "Why is Crude Bullish?",
    category: "ai-query",
    keywords: ["oil", "analysis", "reason"],
    icon: "help-circle",
    action: { type: "ai-query", query: "why_crude_bullish" },
  },
  {
    id: "ai-compare-assets",
    label: "Compare Assets",
    category: "ai-query",
    keywords: ["versus", "vs", "comparison"],
    icon: "git-compare",
    action: { type: "ai-query", query: "compare_assets" },
  },
  {
    id: "ai-market-summary",
    label: "Market Summary",
    category: "ai-query",
    keywords: ["overview", "analysis", "today"],
    icon: "file-text",
    action: { type: "ai-query", query: "market_summary" },
  },
  {
    id: "ai-risk-assessment",
    label: "Risk Assessment",
    category: "ai-query",
    keywords: ["danger", "warning", "exposure"],
    icon: "alert-triangle",
    action: { type: "ai-query", query: "risk_assessment" },
  },

  // Filter commands
  {
    id: "filter-bullish",
    label: "Show Only Bullish",
    category: "filter",
    keywords: ["long", "up", "positive"],
    icon: "trending-up",
    action: { type: "filter", filter: { direction: "bullish" } },
  },
  {
    id: "filter-bearish",
    label: "Show Only Bearish",
    category: "filter",
    keywords: ["short", "down", "negative"],
    icon: "trending-down",
    action: { type: "filter", filter: { direction: "bearish" } },
  },
  {
    id: "filter-high-conviction",
    label: "High Conviction Only",
    category: "filter",
    keywords: ["confident", "strong", "certain"],
    icon: "star",
    action: { type: "filter", filter: { conviction: "high" } },
  },
  {
    id: "filter-energy",
    label: "Filter Energy Assets",
    category: "filter",
    keywords: ["oil", "gas", "crude"],
    icon: "zap",
    action: { type: "filter", filter: { category: "energy" } },
  },
  {
    id: "filter-metals",
    label: "Filter Metals",
    category: "filter",
    keywords: ["gold", "silver", "copper"],
    icon: "gem",
    action: { type: "filter", filter: { category: "metals" } },
  },
  {
    id: "filter-clear",
    label: "Clear All Filters",
    category: "filter",
    keywords: ["reset", "remove", "all"],
    icon: "x",
    action: { type: "filter", filter: {} },
  },
];

export function groupCommandsByCategory(commands: Command[]): CommandGroup[] {
  const groups: Map<CommandCategory, Command[]> = new Map();

  for (const command of commands) {
    const existing = groups.get(command.category) || [];
    existing.push(command);
    groups.set(command.category, existing);
  }

  return Array.from(groups.entries()).map(([category, commands]) => ({
    category,
    label: COMMAND_CATEGORIES[category].label,
    commands,
  }));
}
