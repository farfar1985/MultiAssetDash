/**
 * QDT Nexus API Client
 * Typed API client for frontend consumption
 * Connects to Amira's backend API
 */

import type {
  AssetId,
  SignalDirection,
  BacktestResult,
  ModelAccuracy,
  FeatureImportance,
  HistoricalMetrics,
  StrategyId,
} from "@/types";

// ============================================================================
// Configuration
// ============================================================================

/**
 * Base URL for the backend API
 * Uses environment variable in production, falls back to localhost for development
 */
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";

/**
 * API version prefix
 */
const API_VERSION = "/api/v1";

/**
 * Full API base path
 */
const getApiUrl = (endpoint: string) => `${API_BASE_URL}${API_VERSION}${endpoint}`;

// ============================================================================
// Types
// ============================================================================

export type Horizon = "D+1" | "D+5" | "D+10";

export type EnsembleMethod =
  | "accuracy_weighted"
  | "exponential_decay"
  | "top_k_sharpe"
  | "ridge_stacking"
  | "inverse_variance"
  | "pairwise_slope";

export interface AssetData {
  id: AssetId;
  name: string;
  symbol: string;
  category: "energy" | "metals" | "crypto" | "agriculture";
  currentPrice: number;
  change24h: number;
  changePercent24h: number;
}

export interface SignalData {
  assetId: AssetId;
  direction: SignalDirection;
  confidence: number;
  horizon: Horizon;
  modelsAgreeing: number;
  modelsTotal: number;
  sharpeRatio: number;
  directionalAccuracy: number;
  totalReturn: number;
  generatedAt: string;
}

export interface PerformanceMetrics {
  sharpeRatio: number;
  directionalAccuracy: number;
  totalReturn: number;
  maxDrawdown: number;
  winRate: number;
  modelCount: number;
  lastUpdated: string;
}

export interface EnsembleResult {
  method: EnsembleMethod;
  signal: SignalData;
  modelWeights: Record<string, number>;
  backtestMetrics: {
    sharpeRatio: number;
    directionalAccuracy: number;
    totalReturn: number;
    maxDrawdown: number;
  };
}

export interface ModelAgreement {
  bullishCount: number;
  bearishCount: number;
  neutralCount: number;
  totalModels: number;
  overallDirection: SignalDirection;
  topModels: Array<{
    name: string;
    direction: SignalDirection;
    confidence: number;
    weight: number;
  }>;
}

export interface HistoricalDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  signal?: SignalDirection;
  confidence?: number;
}

export interface PortfolioSummary {
  ensembleMethod: EnsembleMethod;
  overallSignal: SignalDirection;
  overallConfidence: number;
  totalModels: number;
  lastUpdated: string;
  assetBreakdown: Array<{
    assetId: AssetId;
    direction: SignalDirection;
    confidence: number;
  }>;
}

// Backend API specific types
export interface BackendAsset {
  name: string;
  id: string;
  display_name: string;
}

export interface BackendSignal {
  date: string;
  signal: "LONG" | "SHORT" | "NEUTRAL";
  net_prob: number;
  strength: number;
}

export interface BackendForecast {
  forecasts: Array<{
    horizon: string;
    probability_up: number;
    probability_down: number;
  }>;
  signal: string;
  confidence: number;
  viable_horizons: string[];
  timestamp: string;
}

export interface BackendMetrics {
  asset: string;
  optimized_metrics: {
    total_return: number;
    sharpe_ratio: number;
    win_rate: number;
    profit_factor: number;
    max_drawdown: number;
    total_trades: number;
  };
  raw_metrics: {
    total_return: number;
    sharpe_ratio: number;
    win_rate: number;
    profit_factor: number;
    max_drawdown: number;
    total_trades: number;
  };
  configuration: {
    viable_horizons: string[];
    threshold: number;
    avg_accuracy: number;
    health_score: number;
  };
  timestamp: string;
}

export interface EnsembleConfig {
  viable_horizons: string[];
  threshold: number;
  avg_accuracy: number;
  health_score: number;
}

export interface HealthStatus {
  success: boolean;
  service: string;
  version: string;
  timestamp: string;
}

// API Response wrapper
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
  // Backend responses include data at root level
  [key: string]: unknown;
}

// ============================================================================
// API Error
// ============================================================================

export class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public code?: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ============================================================================
// API Key Management
// ============================================================================

let apiKey: string | null = null;

/**
 * Set the API key for authenticated requests
 */
export function setApiKey(key: string) {
  apiKey = key;
}

/**
 * Get the current API key
 */
export function getApiKey(): string | null {
  // Check in order: module-level variable, env var, localStorage
  return apiKey
    || process.env.NEXT_PUBLIC_API_KEY
    || (typeof window !== "undefined" ? localStorage.getItem("qdt_api_key") : null);
}

/**
 * Clear the stored API key
 */
export function clearApiKey() {
  apiKey = null;
  if (typeof window !== "undefined") {
    localStorage.removeItem("qdt_api_key");
  }
}

// ============================================================================
// Fetch wrapper with error handling
// ============================================================================

async function fetchApi<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  const currentApiKey = getApiKey();

  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...options?.headers,
  };

  // Add API key if available
  if (currentApiKey) {
    (headers as Record<string, string>)["X-API-Key"] = currentApiKey;
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.error || `HTTP error ${response.status}`,
      response.status,
      errorData.code
    );
  }

  const result = await response.json();

  // Backend wraps responses with success flag
  if (result.success === false) {
    throw new ApiError(result.error || "Unknown error");
  }

  return result;
}

// ============================================================================
// Legacy API Client (for backward compatibility with mock endpoints)
// ============================================================================

async function fetchLegacyApi<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      errorData.error || `HTTP error ${response.status}`,
      response.status,
      errorData.code
    );
  }

  const result: ApiResponse<T> = await response.json();

  if (!result.success) {
    throw new ApiError(result.error || "Unknown error");
  }

  return result.data as T;
}

// ============================================================================
// Backend API Client (Amira's API on port 5001)
// ============================================================================

export const backendApi = {
  /**
   * Health check - verify connection to backend
   */
  health: (): Promise<HealthStatus> =>
    fetchApi<HealthStatus>(getApiUrl("/health")),

  /**
   * Get list of available assets
   */
  getAssets: async (): Promise<BackendAsset[]> => {
    const response = await fetchApi<{ assets: BackendAsset[]; count: number }>(
      getApiUrl("/assets")
    );
    return response.assets;
  },

  /**
   * Get signal data for an asset with specified horizons
   * @param asset - Asset name (e.g., "crude-oil", "gold")
   * @param horizons - Comma-separated horizons (e.g., "D+5,D+7,D+10")
   */
  getSignal: async (
    asset: string,
    horizons?: string
  ): Promise<{ data: BackendSignal[]; count: number }> => {
    const params = new URLSearchParams();
    if (horizons) params.set("horizons", horizons);
    const query = params.toString() ? `?${params.toString()}` : "";
    return fetchApi(getApiUrl(`/signals/${asset}${query}`));
  },

  /**
   * Get live forecast for an asset
   */
  getForecast: (asset: string): Promise<BackendForecast> =>
    fetchApi<BackendForecast>(getApiUrl(`/forecast/${asset}`)),

  /**
   * Get performance metrics for an asset
   */
  getMetrics: (asset: string): Promise<BackendMetrics> =>
    fetchApi<BackendMetrics>(getApiUrl(`/metrics/${asset}`)),

  /**
   * Get OHLCV data for an asset
   */
  getOHLCV: async (
    asset: string,
    startDate?: string,
    endDate?: string
  ): Promise<HistoricalDataPoint[]> => {
    const params = new URLSearchParams();
    if (startDate) params.set("start_date", startDate);
    if (endDate) params.set("end_date", endDate);
    const query = params.toString() ? `?${params.toString()}` : "";
    const response = await fetchApi<{ data: HistoricalDataPoint[]; count: number }>(
      getApiUrl(`/ohlcv/${asset}${query}`)
    );
    return response.data;
  },

  /**
   * Get equity curve for an asset
   */
  getEquity: async (asset: string): Promise<{ equity_curve: Array<{ date: string; equity: number }>; final_equity: number; total_return: number }> => {
    return fetchApi(getApiUrl(`/equity/${asset}`));
  },

  /**
   * Get confidence stats for an asset
   */
  getConfidence: (asset: string): Promise<{ stats_by_horizon: Record<string, unknown>; overall_stats: Record<string, unknown> }> =>
    fetchApi(getApiUrl(`/confidence/${asset}`)),

  /**
   * Get ensemble configuration from metrics
   */
  getEnsembleConfig: async (asset: string): Promise<EnsembleConfig> => {
    const metrics = await backendApi.getMetrics(asset);
    return metrics.configuration;
  },

  /**
   * Get strategy configuration for an asset
   * @param asset - Asset name (e.g., "crude-oil")
   * @param strategy - Strategy name (e.g., "vol70_consec3", "optimal")
   */
  getConfig: async (asset: string, strategy: string = "optimal"): Promise<{
    asset: string;
    strategy: string;
    config: Record<string, unknown>;
  }> => {
    const params = new URLSearchParams();
    if (strategy !== "optimal") params.set("strategy", strategy);
    const query = params.toString() ? `?${params.toString()}` : "";
    return fetchApi(getApiUrl(`/config/${asset}${query}`));
  },

  /**
   * List all available strategy configurations
   */
  listConfigs: async (): Promise<{
    configs: Array<{
      filename: string;
      strategy: string;
      asset: string;
      viable_horizons: number[];
      sharpe_ratio: number;
      win_rate: number;
    }>;
    count: number;
  }> => {
    return fetchApi(getApiUrl("/configs"));
  },
};

// ============================================================================
// API Client (Legacy mock endpoints for backward compatibility)
// ============================================================================

// Re-export quantum_ml types for convenience
export type {
  BacktestResult,
  ModelAccuracy,
  FeatureImportance,
  HistoricalMetrics,
  StrategyId,
} from "@/types";

export const api = {
  /**
   * Get all available assets
   */
  getAssets: (): Promise<AssetData[]> =>
    fetchLegacyApi<AssetData[]>("/api/assets"),

  /**
   * Get details for a specific asset
   */
  getAsset: (symbol: string): Promise<AssetData> =>
    fetchLegacyApi<AssetData>(`/api/assets/${symbol}`),

  /**
   * Get signal for an asset at a specific horizon with optional ensemble method
   */
  getSignal: (
    symbol: string,
    horizon: Horizon = "D+1",
    method?: EnsembleMethod
  ): Promise<SignalData> => {
    const params = new URLSearchParams({ horizon });
    if (method) params.set("method", method);
    return fetchLegacyApi<SignalData>(`/api/signals/${symbol}?${params.toString()}`);
  },

  /**
   * Get performance metrics for an asset with optional ensemble method
   */
  getMetrics: (
    symbol: string,
    method?: EnsembleMethod
  ): Promise<PerformanceMetrics> => {
    const url = method
      ? `/api/metrics/${symbol}?method=${method}`
      : `/api/metrics/${symbol}`;
    return fetchLegacyApi<PerformanceMetrics>(url);
  },

  /**
   * Get ensemble results for an asset with a specific method
   */
  getEnsembleResults: (
    symbol: string,
    method: EnsembleMethod = "accuracy_weighted"
  ): Promise<EnsembleResult> =>
    fetchLegacyApi<EnsembleResult>(`/api/ensemble/${symbol}?method=${method}`),

  /**
   * Get historical price and signal data
   */
  getHistorical: (
    symbol: string,
    startDate: string,
    endDate: string
  ): Promise<HistoricalDataPoint[]> =>
    fetchLegacyApi<HistoricalDataPoint[]>(
      `/api/historical/${symbol}?start=${startDate}&end=${endDate}`
    ),

  /**
   * Get model agreement breakdown for an asset
   */
  getModelAgreement: (symbol: string): Promise<ModelAgreement> =>
    fetchLegacyApi<ModelAgreement>(`/api/models/${symbol}/agreement`),

  /**
   * Get portfolio summary across all assets
   */
  getPortfolioSummary: (method?: EnsembleMethod): Promise<PortfolioSummary> =>
    fetchLegacyApi<PortfolioSummary>(
      `/api/portfolio/summary${method ? `?method=${method}` : ""}`
    ),

  // ==========================================================================
  // Quantum ML Endpoints
  // ==========================================================================

  /**
   * Get backtest results for a symbol and strategy
   * Returns official quantum_ml performance metrics
   */
  getBacktest: (
    symbol: string,
    strategy: StrategyId = "ensemble_weighted"
  ): Promise<BacktestResult> =>
    fetchLegacyApi<BacktestResult>(
      `/api/quantum-ml/backtest/${symbol}?strategy=${strategy}`
    ),

  /**
   * Get model accuracy metrics for a symbol
   * Returns accuracy breakdown for all models in the ensemble
   */
  getModelAccuracy: (symbol: string): Promise<ModelAccuracy[]> =>
    fetchLegacyApi<ModelAccuracy[]>(`/api/quantum-ml/accuracy/${symbol}`),

  /**
   * Get historical metrics for time-travel analysis
   * Returns backtest snapshots within the specified date range
   */
  getHistoricalMetrics: (
    symbol: string,
    startDate: string,
    endDate: string
  ): Promise<HistoricalMetrics[]> =>
    fetchLegacyApi<HistoricalMetrics[]>(
      `/api/quantum-ml/historical/${symbol}?start=${startDate}&end=${endDate}`
    ),

  /**
   * Get feature importance rankings for a symbol
   * Returns features sorted by importance score
   */
  getFeatureImportance: (symbol: string): Promise<FeatureImportance[]> =>
    fetchLegacyApi<FeatureImportance[]>(`/api/quantum-ml/features/${symbol}`),
};

export default api;
