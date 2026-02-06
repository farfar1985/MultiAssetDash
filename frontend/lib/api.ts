/**
 * QDT Nexus API Integration Layer
 *
 * This module provides typed API functions for the frontend.
 * Currently uses mock data - swap USE_MOCK_DATA to false and
 * implement the fetch calls to connect to the real backend.
 */

import type {
  ApiResponse,
  PaginatedResponse,
  Signal,
  DashboardMetrics,
  PersonaId,
  AssetId,
} from "@/types";

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";
const API_VERSION = "/api/v1";

/**
 * Toggle this to switch between mock data and real API calls
 */
const USE_MOCK_DATA = true;

/**
 * Build full API URL for an endpoint
 */
const getApiUrl = (endpoint: string) => `${API_BASE_URL}${API_VERSION}${endpoint}`;

// ============================================================================
// Types - Quantum API
// ============================================================================

export type QuantumRegime = "LOW_VOL" | "NORMAL" | "ELEVATED" | "CRISIS";
export type ContagionLevel = "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
export type SignalDirection = "LONG" | "SHORT" | "NEUTRAL";
export type Horizon = "D+1" | "D+5" | "D+7" | "D+10" | "D+14" | "D+21";

export interface QuantumRegimeData {
  assetId: AssetId;
  regime: QuantumRegime;
  confidence: number;
  realizedVol: number;
  impliedVol: number;
  volSpread: number;
  regimeStartDate: string;
  daysInRegime: number;
  previousRegime: QuantumRegime | null;
  timestamp: string;
}

export interface ContagionStatus {
  level: ContagionLevel;
  score: number;
  correlationMatrix: Record<string, Record<string, number>>;
  highestRiskPair: [AssetId, AssetId] | null;
  highestRiskCorrelation: number;
  entanglementScore: number;
  systemicRiskIndex: number;
  alerts: ContagionAlert[];
  timestamp: string;
}

export interface ContagionAlert {
  severity: "warning" | "critical";
  message: string;
  assets: AssetId[];
  correlation: number;
}

export interface OptimalHorizon {
  horizon: Horizon;
  accuracy: number;
  sharpeRatio: number;
  winRate: number;
  avgReturn: number;
  isOptimal: boolean;
  sampleSize: number;
}

export interface OptimalHorizonsData {
  assetId: AssetId;
  horizons: OptimalHorizon[];
  recommendedHorizon: Horizon;
  viableHorizons: Horizon[];
  timestamp: string;
}

export interface SignalData {
  assetId: AssetId;
  direction: SignalDirection;
  confidence: number;
  horizon: Horizon;
  probability: number;
  strength: number;
  modelAgreement: number;
  timestamp: string;
}

export interface AssetSignals {
  assetId: AssetId;
  currentSignal: SignalData;
  historicalSignals: SignalData[];
  performance: {
    accuracy: number;
    sharpeRatio: number;
    totalReturn: number;
  };
  timestamp: string;
}

// ============================================================================
// Mock Data
// ============================================================================

const MOCK_REGIMES: Record<AssetId, QuantumRegimeData> = {
  "crude-oil": {
    assetId: "crude-oil",
    regime: "ELEVATED",
    confidence: 0.87,
    realizedVol: 0.32,
    impliedVol: 0.38,
    volSpread: 0.06,
    regimeStartDate: "2024-01-15",
    daysInRegime: 12,
    previousRegime: "NORMAL",
    timestamp: new Date().toISOString(),
  },
  "gold": {
    assetId: "gold",
    regime: "LOW_VOL",
    confidence: 0.92,
    realizedVol: 0.12,
    impliedVol: 0.14,
    volSpread: 0.02,
    regimeStartDate: "2024-01-08",
    daysInRegime: 19,
    previousRegime: "NORMAL",
    timestamp: new Date().toISOString(),
  },
  "natural-gas": {
    assetId: "natural-gas",
    regime: "CRISIS",
    confidence: 0.78,
    realizedVol: 0.58,
    impliedVol: 0.72,
    volSpread: 0.14,
    regimeStartDate: "2024-01-22",
    daysInRegime: 5,
    previousRegime: "ELEVATED",
    timestamp: new Date().toISOString(),
  },
  "bitcoin": {
    assetId: "bitcoin",
    regime: "NORMAL",
    confidence: 0.84,
    realizedVol: 0.45,
    impliedVol: 0.48,
    volSpread: 0.03,
    regimeStartDate: "2024-01-10",
    daysInRegime: 17,
    previousRegime: "ELEVATED",
    timestamp: new Date().toISOString(),
  },
  "silver": {
    assetId: "silver",
    regime: "NORMAL",
    confidence: 0.89,
    realizedVol: 0.22,
    impliedVol: 0.25,
    volSpread: 0.03,
    regimeStartDate: "2024-01-12",
    daysInRegime: 15,
    previousRegime: "LOW_VOL",
    timestamp: new Date().toISOString(),
  },
  "copper": {
    assetId: "copper",
    regime: "NORMAL",
    confidence: 0.85,
    realizedVol: 0.28,
    impliedVol: 0.30,
    volSpread: 0.02,
    regimeStartDate: "2024-01-14",
    daysInRegime: 13,
    previousRegime: "LOW_VOL",
    timestamp: new Date().toISOString(),
  },
  "wheat": {
    assetId: "wheat",
    regime: "ELEVATED",
    confidence: 0.72,
    realizedVol: 0.35,
    impliedVol: 0.42,
    volSpread: 0.07,
    regimeStartDate: "2024-01-18",
    daysInRegime: 9,
    previousRegime: "NORMAL",
    timestamp: new Date().toISOString(),
  },
  "corn": {
    assetId: "corn",
    regime: "NORMAL",
    confidence: 0.88,
    realizedVol: 0.24,
    impliedVol: 0.26,
    volSpread: 0.02,
    regimeStartDate: "2024-01-11",
    daysInRegime: 16,
    previousRegime: "LOW_VOL",
    timestamp: new Date().toISOString(),
  },
  "soybean": {
    assetId: "soybean",
    regime: "LOW_VOL",
    confidence: 0.91,
    realizedVol: 0.18,
    impliedVol: 0.20,
    volSpread: 0.02,
    regimeStartDate: "2024-01-09",
    daysInRegime: 18,
    previousRegime: "NORMAL",
    timestamp: new Date().toISOString(),
  },
  "platinum": {
    assetId: "platinum",
    regime: "NORMAL",
    confidence: 0.83,
    realizedVol: 0.25,
    impliedVol: 0.28,
    volSpread: 0.03,
    regimeStartDate: "2024-01-13",
    daysInRegime: 14,
    previousRegime: "ELEVATED",
    timestamp: new Date().toISOString(),
  },
};

const MOCK_CONTAGION: ContagionStatus = {
  level: "MODERATE",
  score: 0.62,
  correlationMatrix: {
    "crude-oil": { "gold": 0.35, "natural-gas": 0.72, "bitcoin": 0.28 },
    "gold": { "crude-oil": 0.35, "silver": 0.89, "bitcoin": 0.15 },
    "natural-gas": { "crude-oil": 0.72, "gold": 0.21, "bitcoin": 0.18 },
    "bitcoin": { "gold": 0.15, "crude-oil": 0.28, "silver": 0.42 },
  },
  highestRiskPair: ["gold", "silver"],
  highestRiskCorrelation: 0.89,
  entanglementScore: 0.58,
  systemicRiskIndex: 0.45,
  alerts: [
    {
      severity: "warning",
      message: "Metals correlation spike detected",
      assets: ["gold", "silver"],
      correlation: 0.89,
    },
    {
      severity: "warning",
      message: "Energy sector showing elevated correlation",
      assets: ["crude-oil", "natural-gas"],
      correlation: 0.72,
    },
  ],
  timestamp: new Date().toISOString(),
};

const MOCK_OPTIMAL_HORIZONS: Partial<Record<AssetId, OptimalHorizonsData>> = {
  "crude-oil": {
    assetId: "crude-oil",
    horizons: [
      { horizon: "D+1", accuracy: 0.52, sharpeRatio: 0.8, winRate: 0.51, avgReturn: 0.002, isOptimal: false, sampleSize: 250 },
      { horizon: "D+5", accuracy: 0.61, sharpeRatio: 1.4, winRate: 0.58, avgReturn: 0.008, isOptimal: true, sampleSize: 250 },
      { horizon: "D+7", accuracy: 0.58, sharpeRatio: 1.2, winRate: 0.56, avgReturn: 0.010, isOptimal: false, sampleSize: 250 },
      { horizon: "D+10", accuracy: 0.55, sharpeRatio: 1.0, winRate: 0.54, avgReturn: 0.012, isOptimal: false, sampleSize: 250 },
    ],
    recommendedHorizon: "D+5",
    viableHorizons: ["D+5", "D+7"],
    timestamp: new Date().toISOString(),
  },
  "gold": {
    assetId: "gold",
    horizons: [
      { horizon: "D+1", accuracy: 0.54, sharpeRatio: 0.9, winRate: 0.53, avgReturn: 0.001, isOptimal: false, sampleSize: 250 },
      { horizon: "D+5", accuracy: 0.59, sharpeRatio: 1.3, winRate: 0.57, avgReturn: 0.004, isOptimal: false, sampleSize: 250 },
      { horizon: "D+7", accuracy: 0.63, sharpeRatio: 1.6, winRate: 0.60, avgReturn: 0.006, isOptimal: true, sampleSize: 250 },
      { horizon: "D+10", accuracy: 0.60, sharpeRatio: 1.4, winRate: 0.58, avgReturn: 0.008, isOptimal: false, sampleSize: 250 },
    ],
    recommendedHorizon: "D+7",
    viableHorizons: ["D+5", "D+7", "D+10"],
    timestamp: new Date().toISOString(),
  },
  "natural-gas": {
    assetId: "natural-gas",
    horizons: [
      { horizon: "D+1", accuracy: 0.48, sharpeRatio: 0.4, winRate: 0.48, avgReturn: -0.001, isOptimal: false, sampleSize: 250 },
      { horizon: "D+5", accuracy: 0.56, sharpeRatio: 1.1, winRate: 0.55, avgReturn: 0.012, isOptimal: true, sampleSize: 250 },
      { horizon: "D+7", accuracy: 0.53, sharpeRatio: 0.9, winRate: 0.52, avgReturn: 0.015, isOptimal: false, sampleSize: 250 },
      { horizon: "D+10", accuracy: 0.50, sharpeRatio: 0.6, winRate: 0.50, avgReturn: 0.018, isOptimal: false, sampleSize: 250 },
    ],
    recommendedHorizon: "D+5",
    viableHorizons: ["D+5"],
    timestamp: new Date().toISOString(),
  },
  "bitcoin": {
    assetId: "bitcoin",
    horizons: [
      { horizon: "D+1", accuracy: 0.51, sharpeRatio: 0.7, winRate: 0.50, avgReturn: 0.003, isOptimal: false, sampleSize: 250 },
      { horizon: "D+5", accuracy: 0.58, sharpeRatio: 1.2, winRate: 0.56, avgReturn: 0.015, isOptimal: false, sampleSize: 250 },
      { horizon: "D+7", accuracy: 0.62, sharpeRatio: 1.5, winRate: 0.59, avgReturn: 0.020, isOptimal: true, sampleSize: 250 },
      { horizon: "D+10", accuracy: 0.59, sharpeRatio: 1.3, winRate: 0.57, avgReturn: 0.025, isOptimal: false, sampleSize: 250 },
    ],
    recommendedHorizon: "D+7",
    viableHorizons: ["D+5", "D+7", "D+10"],
    timestamp: new Date().toISOString(),
  },
  "silver": {
    assetId: "silver",
    horizons: [
      { horizon: "D+1", accuracy: 0.53, sharpeRatio: 0.85, winRate: 0.52, avgReturn: 0.001, isOptimal: false, sampleSize: 250 },
      { horizon: "D+5", accuracy: 0.60, sharpeRatio: 1.35, winRate: 0.58, avgReturn: 0.005, isOptimal: true, sampleSize: 250 },
      { horizon: "D+7", accuracy: 0.58, sharpeRatio: 1.25, winRate: 0.56, avgReturn: 0.007, isOptimal: false, sampleSize: 250 },
      { horizon: "D+10", accuracy: 0.55, sharpeRatio: 1.1, winRate: 0.54, avgReturn: 0.009, isOptimal: false, sampleSize: 250 },
    ],
    recommendedHorizon: "D+5",
    viableHorizons: ["D+5", "D+7"],
    timestamp: new Date().toISOString(),
  },
};

const MOCK_SIGNALS: Partial<Record<AssetId, AssetSignals>> = {
  "crude-oil": {
    assetId: "crude-oil",
    currentSignal: {
      assetId: "crude-oil",
      direction: "LONG",
      confidence: 0.72,
      horizon: "D+5",
      probability: 0.68,
      strength: 0.75,
      modelAgreement: 0.80,
      timestamp: new Date().toISOString(),
    },
    historicalSignals: [],
    performance: { accuracy: 0.61, sharpeRatio: 1.4, totalReturn: 0.18 },
    timestamp: new Date().toISOString(),
  },
  "gold": {
    assetId: "gold",
    currentSignal: {
      assetId: "gold",
      direction: "NEUTRAL",
      confidence: 0.55,
      horizon: "D+7",
      probability: 0.52,
      strength: 0.45,
      modelAgreement: 0.60,
      timestamp: new Date().toISOString(),
    },
    historicalSignals: [],
    performance: { accuracy: 0.63, sharpeRatio: 1.6, totalReturn: 0.12 },
    timestamp: new Date().toISOString(),
  },
  "natural-gas": {
    assetId: "natural-gas",
    currentSignal: {
      assetId: "natural-gas",
      direction: "SHORT",
      confidence: 0.81,
      horizon: "D+5",
      probability: 0.78,
      strength: 0.85,
      modelAgreement: 0.88,
      timestamp: new Date().toISOString(),
    },
    historicalSignals: [],
    performance: { accuracy: 0.56, sharpeRatio: 1.1, totalReturn: 0.22 },
    timestamp: new Date().toISOString(),
  },
  "bitcoin": {
    assetId: "bitcoin",
    currentSignal: {
      assetId: "bitcoin",
      direction: "LONG",
      confidence: 0.68,
      horizon: "D+7",
      probability: 0.65,
      strength: 0.70,
      modelAgreement: 0.75,
      timestamp: new Date().toISOString(),
    },
    historicalSignals: [],
    performance: { accuracy: 0.62, sharpeRatio: 1.5, totalReturn: 0.35 },
    timestamp: new Date().toISOString(),
  },
  "silver": {
    assetId: "silver",
    currentSignal: {
      assetId: "silver",
      direction: "LONG",
      confidence: 0.65,
      horizon: "D+5",
      probability: 0.62,
      strength: 0.68,
      modelAgreement: 0.72,
      timestamp: new Date().toISOString(),
    },
    historicalSignals: [],
    performance: { accuracy: 0.60, sharpeRatio: 1.35, totalReturn: 0.14 },
    timestamp: new Date().toISOString(),
  },
};

// ============================================================================
// API Functions - Quantum Endpoints
// ============================================================================

/**
 * Get quantum volatility regime for an asset
 *
 * @param assetId - The asset identifier (e.g., "crude-oil", "gold")
 * @returns Promise<QuantumRegimeData> - Regime classification and metrics
 *
 * Real endpoint: GET /api/v1/quantum/regime/{assetId}
 */
export async function getQuantumRegime(assetId: AssetId): Promise<QuantumRegimeData> {
  if (USE_MOCK_DATA) {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 100));

    const regime = MOCK_REGIMES[assetId];
    if (!regime) {
      throw new Error(`Unknown asset: ${assetId}`);
    }
    return { ...regime, timestamp: new Date().toISOString() };
  }

  // Real API call
  const response = await fetch(getApiUrl(`/quantum/regime/${assetId}`));
  if (!response.ok) {
    throw new Error(`Failed to fetch regime for ${assetId}: ${response.status}`);
  }
  return response.json();
}

/**
 * Get cross-asset contagion/correlation status
 *
 * @returns Promise<ContagionStatus> - System-wide contagion metrics
 *
 * Real endpoint: GET /api/v1/quantum/contagion
 */
export async function getContagionStatus(): Promise<ContagionStatus> {
  if (USE_MOCK_DATA) {
    await new Promise(resolve => setTimeout(resolve, 150));
    return { ...MOCK_CONTAGION, timestamp: new Date().toISOString() };
  }

  // Real API call
  const response = await fetch(getApiUrl("/quantum/contagion"));
  if (!response.ok) {
    throw new Error(`Failed to fetch contagion status: ${response.status}`);
  }
  return response.json();
}

/**
 * Get optimal forecast horizons for an asset
 *
 * @param assetId - The asset identifier
 * @returns Promise<OptimalHorizonsData> - Horizon analysis with recommendations
 *
 * Real endpoint: GET /api/v1/quantum/horizons/{assetId}
 */
export async function getOptimalHorizons(assetId: AssetId): Promise<OptimalHorizonsData> {
  if (USE_MOCK_DATA) {
    await new Promise(resolve => setTimeout(resolve, 100));

    const horizons = MOCK_OPTIMAL_HORIZONS[assetId];
    if (!horizons) {
      throw new Error(`Unknown asset: ${assetId}`);
    }
    return { ...horizons, timestamp: new Date().toISOString() };
  }

  // Real API call
  const response = await fetch(getApiUrl(`/quantum/horizons/${assetId}`));
  if (!response.ok) {
    throw new Error(`Failed to fetch horizons for ${assetId}: ${response.status}`);
  }
  return response.json();
}

/**
 * Get trading signals for an asset
 *
 * @param assetId - The asset identifier
 * @returns Promise<AssetSignals> - Current and historical signals
 *
 * Real endpoint: GET /api/v1/signals/{assetId}
 */
export async function getSignals(assetId: AssetId): Promise<AssetSignals> {
  if (USE_MOCK_DATA) {
    await new Promise(resolve => setTimeout(resolve, 100));

    const signals = MOCK_SIGNALS[assetId];
    if (!signals) {
      throw new Error(`Unknown asset: ${assetId}`);
    }
    return { ...signals, timestamp: new Date().toISOString() };
  }

  // Real API call
  const response = await fetch(getApiUrl(`/signals/${assetId}`));
  if (!response.ok) {
    throw new Error(`Failed to fetch signals for ${assetId}: ${response.status}`);
  }
  return response.json();
}

// ============================================================================
// Batch/Convenience Functions
// ============================================================================

/**
 * Get all regime data for multiple assets
 */
export async function getAllRegimes(assetIds: AssetId[]): Promise<Partial<Record<AssetId, QuantumRegimeData>>> {
  const results = await Promise.all(
    assetIds.map(async (id) => {
      const regime = await getQuantumRegime(id);
      return [id, regime] as const;
    })
  );
  return Object.fromEntries(results) as Partial<Record<AssetId, QuantumRegimeData>>;
}

/**
 * Get complete quantum dashboard data
 */
export async function getQuantumDashboard(assetIds: AssetId[]): Promise<{
  regimes: Partial<Record<AssetId, QuantumRegimeData>>;
  contagion: ContagionStatus;
  timestamp: string;
}> {
  const [regimes, contagion] = await Promise.all([
    getAllRegimes(assetIds),
    getContagionStatus(),
  ]);

  return {
    regimes,
    contagion,
    timestamp: new Date().toISOString(),
  };
}

// ============================================================================
// Legacy API Client (for backward compatibility)
// ============================================================================

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;

    const defaultHeaders: HeadersInit = {
      "Content-Type": "application/json",
    };

    const config: RequestInit = {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    };

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return {
        data,
        success: true,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return {
        data: null as T,
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
        timestamp: new Date().toISOString(),
      };
    }
  }

  // Dashboard endpoints
  async getDashboardMetrics(persona: PersonaId): Promise<ApiResponse<DashboardMetrics>> {
    return this.request<DashboardMetrics>(`/api/dashboard/${persona}/metrics`);
  }

  // Signal endpoints
  async getSignals(
    persona: PersonaId,
    params?: {
      asset?: AssetId;
      page?: number;
      pageSize?: number;
    }
  ): Promise<PaginatedResponse<Signal>> {
    const searchParams = new URLSearchParams();
    if (params?.asset) searchParams.set("asset", params.asset);
    if (params?.page) searchParams.set("page", params.page.toString());
    if (params?.pageSize) searchParams.set("pageSize", params.pageSize.toString());

    const query = searchParams.toString();
    const endpoint = `/api/signals/${persona}${query ? `?${query}` : ""}`;

    const response = await this.request<Signal[]>(endpoint);
    return {
      ...response,
      data: response.data || [],
      pagination: {
        page: params?.page || 1,
        pageSize: params?.pageSize || 20,
        total: 0,
        totalPages: 0,
      },
    };
  }

  async getSignalById(id: string): Promise<ApiResponse<Signal>> {
    return this.request<Signal>(`/api/signals/${id}`);
  }

  // Asset endpoints
  async getAssetDetails(
    assetId: AssetId,
    persona: PersonaId
  ): Promise<ApiResponse<{
    asset: AssetId;
    currentPrice: number;
    signals: Signal[];
    metrics: Record<string, number>;
  }>> {
    return this.request(`/api/assets/${assetId}?persona=${persona}`);
  }

  // Health check
  async healthCheck(): Promise<ApiResponse<{ status: string; version: string }>> {
    return this.request("/health");
  }
}

export const apiClient = new ApiClient();
export default apiClient;
