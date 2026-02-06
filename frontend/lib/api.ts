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
import type { RegimeData, MarketRegime } from "@/components/ensemble/RegimeIndicator";

export type { RegimeData, MarketRegime };

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";
const API_VERSION = "/api/v1";

/**
 * Toggle this to switch between mock data and real API calls
 */
const USE_MOCK_DATA = false;

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
// HMM Regime Detection API
// ============================================================================

/**
 * Mock HMM regime data for fallback
 */
const MOCK_HMM_REGIME: Record<AssetId, RegimeData> = {
  "crude-oil": {
    regime: "bull",
    confidence: 0.78,
    probabilities: { bull: 0.78, bear: 0.12, sideways: 0.10 },
    daysInRegime: 8,
    volatility: 24.5,
    trendStrength: 0.42,
    historicalAccuracy: 68.5,
  },
  "gold": {
    regime: "sideways",
    confidence: 0.65,
    probabilities: { bull: 0.25, bear: 0.15, sideways: 0.60 },
    daysInRegime: 12,
    volatility: 14.2,
    trendStrength: 0.08,
    historicalAccuracy: 62.0,
  },
  "bitcoin": {
    regime: "high-volatility",
    confidence: 0.72,
    probabilities: { bull: 0.35, bear: 0.30, sideways: 0.35 },
    daysInRegime: 5,
    volatility: 52.8,
    trendStrength: -0.15,
    historicalAccuracy: 55.6,
  },
  "sp500": {
    regime: "bull",
    confidence: 0.82,
    probabilities: { bull: 0.82, bear: 0.08, sideways: 0.10 },
    daysInRegime: 15,
    volatility: 16.3,
    trendStrength: 0.55,
    historicalAccuracy: 57.4,
  },
};

/**
 * Map backend asset IDs to frontend AssetId strings
 * Updated 2026-02-06 with Amira's 13-asset HMM models
 */
const ASSET_ID_MAP: Record<string, number> = {
  // Core assets with full ensemble support
  "crude-oil": 1866,
  "sp500": 1625,
  "bitcoin": 1860,
  "gold": 1861,
  // Additional assets with HMM regime models
  "natural-gas": 1862,
  "silver": 1863,
  "copper": 1864,
  "wheat": 1865,
  "corn": 1867,
  "soybean": 1868,
  "platinum": 1869,
  "ethereum": 1870,
  "nasdaq": 1871,
};

/**
 * Get HMM-detected market regime for an asset
 *
 * @param assetId - The asset identifier (e.g., "crude-oil", "gold")
 * @returns Promise<RegimeData> - Regime classification matching RegimeIndicator interface
 *
 * Real endpoint: GET /api/v1/hmm/regime/{asset_id}
 */
export async function getHMMRegime(assetId: AssetId): Promise<RegimeData> {
  if (USE_MOCK_DATA) {
    await new Promise(resolve => setTimeout(resolve, 100));

    const regime = MOCK_HMM_REGIME[assetId];
    if (!regime) {
      // Return default mock for unknown assets
      return {
        regime: "sideways",
        confidence: 0.5,
        probabilities: { bull: 0.33, bear: 0.33, sideways: 0.34 },
        daysInRegime: 1,
        volatility: 20.0,
        trendStrength: 0.0,
        historicalAccuracy: 50.0,
      };
    }
    return regime;
  }

  // Real API call - map frontend assetId to backend numeric ID
  const backendId = ASSET_ID_MAP[assetId];
  if (!backendId) {
    throw new Error(`Unknown asset: ${assetId}`);
  }

  const response = await fetch(getApiUrl(`/hmm/regime/${backendId}`));
  if (!response.ok) {
    // Fallback to mock on error
    console.warn(`HMM regime fetch failed for ${assetId}, using mock data`);
    return MOCK_HMM_REGIME[assetId] || {
      regime: "sideways",
      confidence: 0.5,
      probabilities: { bull: 0.33, bear: 0.33, sideways: 0.34 },
      daysInRegime: 1,
      volatility: 20.0,
      trendStrength: 0.0,
    };
  }

  const data = await response.json();

  // Transform backend response to RegimeData interface
  return {
    regime: data.regime as MarketRegime,
    confidence: data.confidence,
    probabilities: data.probabilities,
    daysInRegime: data.daysInRegime,
    volatility: data.volatility,
    trendStrength: data.trendStrength,
    historicalAccuracy: data.historicalAccuracy,
  };
}

/**
 * Get HMM regime data for all configured assets
 */
export async function getAllHMMRegimes(): Promise<Partial<Record<AssetId, RegimeData>>> {
  const assetIds = Object.keys(ASSET_ID_MAP) as AssetId[];
  const results = await Promise.all(
    assetIds.map(async (id) => {
      try {
        const regime = await getHMMRegime(id);
        return [id, regime] as const;
      } catch {
        return [id, null] as const;
      }
    })
  );
  return Object.fromEntries(results.filter(([, v]) => v !== null)) as Partial<Record<AssetId, RegimeData>>;
}

// ============================================================================
// Ensemble Confidence API
// ============================================================================

import type {
  EnsembleConfidenceData,
  ConfidenceWeight,
} from "@/components/ensemble/EnsembleConfidenceCard";
import type { PairwiseVotingData, HorizonPairVote } from "@/components/ensemble/PairwiseVotingChart";
import type { ConfidenceInterval } from "@/components/ensemble/ConfidenceIntervalBar";

export type { EnsembleConfidenceData, ConfidenceWeight, PairwiseVotingData, HorizonPairVote, ConfidenceInterval };

/**
 * Mock ensemble confidence data
 */
const MOCK_ENSEMBLE_CONFIDENCE: EnsembleConfidenceData = {
  confidence: 72,
  direction: "bullish",
  weights: [
    { method: "TopK_Sharpe", weight: 0.35, contribution: 0.45, accuracy: 68.5 },
    { method: "Magnitude", weight: 0.30, contribution: 0.32, accuracy: 62.0 },
    { method: "Recent_Perf", weight: 0.35, contribution: 0.38, accuracy: 65.0 },
  ],
  modelsAgreeing: 58,
  modelsTotal: 80,
  ensembleMethod: "accuracy-weighted",
};

/**
 * Get ensemble confidence data for an asset
 */
export async function getEnsembleConfidence(assetId: AssetId): Promise<EnsembleConfidenceData> {
  if (USE_MOCK_DATA) {
    await new Promise(resolve => setTimeout(resolve, 100));
    return { ...MOCK_ENSEMBLE_CONFIDENCE };
  }

  const backendId = ASSET_ID_MAP[assetId];
  if (!backendId) {
    throw new Error(`Unknown asset: ${assetId}`);
  }

  const response = await fetch(getApiUrl(`/ensemble/confidence/${backendId}`));
  if (!response.ok) {
    console.warn(`Ensemble confidence fetch failed for ${assetId}, using mock data`);
    return { ...MOCK_ENSEMBLE_CONFIDENCE };
  }

  const data = await response.json();
  return {
    confidence: data.confidence,
    direction: data.direction,
    weights: data.weights,
    modelsAgreeing: data.modelsAgreeing,
    modelsTotal: data.modelsTotal,
    ensembleMethod: data.ensembleMethod,
  };
}

// ============================================================================
// Pairwise Voting API
// ============================================================================

/**
 * Mock pairwise voting data
 */
const MOCK_PAIRWISE_VOTING: PairwiseVotingData = {
  votes: [
    { h1: "D+1", h2: "D+5", vote: "bullish", magnitude: 1.25, weight: 0.2 },
    { h1: "D+1", h2: "D+10", vote: "bullish", magnitude: 2.10, weight: 0.2 },
    { h1: "D+5", h2: "D+10", vote: "neutral", magnitude: 0.45, weight: 0.2 },
    { h1: "D+3", h2: "D+7", vote: "bearish", magnitude: 0.82, weight: 0.2 },
    { h1: "D+7", h2: "D+10", vote: "bullish", magnitude: 0.95, weight: 0.2 },
  ],
  bullishCount: 3,
  bearishCount: 1,
  neutralCount: 1,
  netProbability: 0.4,
  signal: "bullish",
};

/**
 * Get pairwise voting data for an asset
 */
export async function getPairwiseVoting(assetId: AssetId): Promise<PairwiseVotingData> {
  if (USE_MOCK_DATA) {
    await new Promise(resolve => setTimeout(resolve, 100));
    return { ...MOCK_PAIRWISE_VOTING };
  }

  const backendId = ASSET_ID_MAP[assetId];
  if (!backendId) {
    throw new Error(`Unknown asset: ${assetId}`);
  }

  const response = await fetch(getApiUrl(`/ensemble/pairwise/${backendId}`));
  if (!response.ok) {
    console.warn(`Pairwise voting fetch failed for ${assetId}, using mock data`);
    return { ...MOCK_PAIRWISE_VOTING };
  }

  const data = await response.json();
  return {
    votes: data.votes,
    bullishCount: data.bullishCount,
    bearishCount: data.bearishCount,
    neutralCount: data.neutralCount,
    netProbability: data.netProbability,
    signal: data.signal,
  };
}

// ============================================================================
// Confidence Interval API
// ============================================================================

/**
 * Mock confidence interval data
 */
const MOCK_CONFIDENCE_INTERVAL: ConfidenceInterval = {
  lower: -0.85,
  point: 1.25,
  upper: 3.35,
  coverage: 0.90,
};

/**
 * Get prediction confidence interval for an asset
 */
export async function getConfidenceInterval(
  assetId: AssetId,
  horizon: number = 5,
  coverage: number = 0.90
): Promise<ConfidenceInterval> {
  if (USE_MOCK_DATA) {
    await new Promise(resolve => setTimeout(resolve, 100));
    return { ...MOCK_CONFIDENCE_INTERVAL };
  }

  const backendId = ASSET_ID_MAP[assetId];
  if (!backendId) {
    throw new Error(`Unknown asset: ${assetId}`);
  }

  const response = await fetch(getApiUrl(`/ensemble/interval/${backendId}?horizon=${horizon}&coverage=${coverage}`));
  if (!response.ok) {
    console.warn(`Confidence interval fetch failed for ${assetId}, using mock data`);
    return { ...MOCK_CONFIDENCE_INTERVAL };
  }

  const data = await response.json();
  return {
    lower: data.lower,
    point: data.point,
    upper: data.upper,
    coverage: data.coverage,
  };
}

// ============================================================================
// Combined Ensemble Dashboard API
// ============================================================================

export interface EnsembleDashboardData {
  regime: RegimeData;
  confidence: EnsembleConfidenceData;
  pairwise: PairwiseVotingData;
  interval: ConfidenceInterval;
  timestamp: string;
}

/**
 * Get all ensemble data for an asset in a single call
 */
export async function getEnsembleDashboard(assetId: AssetId): Promise<EnsembleDashboardData> {
  if (USE_MOCK_DATA) {
    await new Promise(resolve => setTimeout(resolve, 150));
    const regime = await getHMMRegime(assetId);
    return {
      regime,
      confidence: { ...MOCK_ENSEMBLE_CONFIDENCE },
      pairwise: { ...MOCK_PAIRWISE_VOTING },
      interval: { ...MOCK_CONFIDENCE_INTERVAL },
      timestamp: new Date().toISOString(),
    };
  }

  const backendId = ASSET_ID_MAP[assetId];
  if (!backendId) {
    throw new Error(`Unknown asset: ${assetId}`);
  }

  const response = await fetch(getApiUrl(`/ensemble/dashboard/${backendId}`));
  if (!response.ok) {
    // Fallback to individual calls
    const [regime, confidence, pairwise, interval] = await Promise.all([
      getHMMRegime(assetId),
      getEnsembleConfidence(assetId),
      getPairwiseVoting(assetId),
      getConfidenceInterval(assetId),
    ]);
    return { regime, confidence, pairwise, interval, timestamp: new Date().toISOString() };
  }

  const data = await response.json();
  return {
    regime: {
      regime: data.regime.regime,
      confidence: data.regime.confidence,
      probabilities: data.regime.probabilities,
      daysInRegime: data.regime.daysInRegime,
      volatility: data.regime.volatility,
      trendStrength: data.regime.trendStrength,
      historicalAccuracy: data.regime.historicalAccuracy,
    },
    confidence: data.confidence,
    pairwise: data.pairwise,
    interval: data.interval,
    timestamp: data.timestamp,
  };
}

// ============================================================================
// Tier Comparison API
// ============================================================================

export interface TierPrediction {
  signal: "BULLISH" | "BEARISH" | "NEUTRAL";
  confidence: number;
  netProbability?: number;
  weights?: Record<string, number>;
  metadata?: Record<string, unknown>;
  // Tier 2 specific
  uncertainty?: number;
  interval?: { lower: number; upper: number };
  regime?: string;
  // Tier 3 specific
  quantiles?: Record<string, number>;
  attentionWeights?: Record<string, number>;
  explorationBonus?: number;
}

export interface TierConsensus {
  signal: "BULLISH" | "BEARISH" | "NEUTRAL";
  agreement: number;
  tiersAgreeing: number;
  totalTiers: number;
}

export interface TierComparisonData {
  asset_id: number;
  asset_name: string;
  timestamp: string;
  tier1: TierPrediction;
  tier2: TierPrediction;
  tier3: TierPrediction;
  consensus: TierConsensus;
}

/**
 * Mock tier comparison data
 */
const MOCK_TIER_COMPARISON: TierComparisonData = {
  asset_id: 1866,
  asset_name: "Crude_Oil",
  timestamp: new Date().toISOString(),
  tier1: {
    signal: "BULLISH",
    confidence: 0.72,
    netProbability: 0.45,
    weights: {
      accuracy_weighted: 0.35,
      magnitude_weighted: 0.30,
      correlation_weighted: 0.35,
    },
    metadata: { modelsUsed: 45, topHorizons: [9, 10] },
  },
  tier2: {
    signal: "BULLISH",
    confidence: 0.68,
    netProbability: 0.38,
    uncertainty: 0.15,
    interval: { lower: -0.85, upper: 3.35 },
    regime: "bull",
  },
  tier3: {
    signal: "NEUTRAL",
    confidence: 0.55,
    netProbability: 0.12,
    quantiles: { "0.1": -1.2, "0.25": -0.4, "0.5": 0.8, "0.75": 2.1, "0.9": 3.5 },
    attentionWeights: { "h9_h10": 0.42, "h8_h9": 0.28, "h7_h8": 0.18, "h5_h7": 0.12 },
    explorationBonus: 0.08,
  },
  consensus: {
    signal: "BULLISH",
    agreement: 0.67,
    tiersAgreeing: 2,
    totalTiers: 3,
  },
};

/**
 * Get tier comparison data for an asset
 */
export async function getTierComparison(assetId: AssetId): Promise<TierComparisonData> {
  if (USE_MOCK_DATA) {
    await new Promise(resolve => setTimeout(resolve, 100));
    return { ...MOCK_TIER_COMPARISON };
  }

  const backendId = ASSET_ID_MAP[assetId];
  if (!backendId) {
    throw new Error(`Unknown asset: ${assetId}`);
  }

  const response = await fetch(getApiUrl(`/ensemble/tiers/${backendId}`));
  if (!response.ok) {
    console.warn(`Tier comparison fetch failed for ${assetId}, using mock data`);
    return { ...MOCK_TIER_COMPARISON };
  }

  return response.json();
}

// ============================================================================
// Individual Tier Ensemble API
// ============================================================================

export type Tier1Method = "combined" | "accuracy" | "magnitude" | "correlation";
export type Tier2Method = "combined" | "bma" | "regime" | "conformal";
export type Tier3Method = "combined" | "thompson" | "attention" | "quantile";

export interface Tier1Result {
  signal: "BULLISH" | "BEARISH" | "NEUTRAL";
  confidence: number;
  netProbability: number;
  weights: Record<string, number>;
  method: Tier1Method;
  metadata: {
    modelsUsed: number;
    topHorizons: number[];
    timestamp: string;
  };
}

export interface Tier2Result {
  signal: "BULLISH" | "BEARISH" | "NEUTRAL";
  confidence: number;
  netProbability: number;
  uncertainty: number;
  interval: { lower: number; upper: number };
  regime: string;
  method: Tier2Method;
  metadata: {
    modelsUsed: number;
    timestamp: string;
  };
}

export interface Tier3Result {
  signal: "BULLISH" | "BEARISH" | "NEUTRAL";
  confidence: number;
  netProbability: number;
  quantiles: Record<string, number>;
  attentionWeights?: Record<string, number>;
  explorationBonus: number;
  method: Tier3Method;
  metadata: {
    modelsUsed: number;
    timestamp: string;
  };
}

/**
 * Mock data for individual tiers
 */
const MOCK_TIER1: Tier1Result = {
  signal: "BULLISH",
  confidence: 0.72,
  netProbability: 0.45,
  weights: {
    accuracy_weighted: 0.35,
    magnitude_weighted: 0.30,
    correlation_weighted: 0.35,
  },
  method: "combined",
  metadata: { modelsUsed: 45, topHorizons: [9, 10], timestamp: new Date().toISOString() },
};

const MOCK_TIER2: Tier2Result = {
  signal: "BULLISH",
  confidence: 0.68,
  netProbability: 0.38,
  uncertainty: 0.15,
  interval: { lower: -0.85, upper: 3.35 },
  regime: "bull",
  method: "combined",
  metadata: { modelsUsed: 45, timestamp: new Date().toISOString() },
};

const MOCK_TIER3: Tier3Result = {
  signal: "NEUTRAL",
  confidence: 0.55,
  netProbability: 0.12,
  quantiles: { "0.1": -1.2, "0.25": -0.4, "0.5": 0.8, "0.75": 2.1, "0.9": 3.5 },
  attentionWeights: { "h9_h10": 0.42, "h8_h9": 0.28, "h7_h8": 0.18, "h5_h7": 0.12 },
  explorationBonus: 0.08,
  method: "combined",
  metadata: { modelsUsed: 45, timestamp: new Date().toISOString() },
};

/**
 * Get Tier 1 ensemble prediction for an asset
 * Tier 1 methods: accuracy-weighted, magnitude-weighted, error correlation
 */
export async function getTier1Ensemble(
  assetId: AssetId,
  method: Tier1Method = "combined"
): Promise<Tier1Result> {
  if (USE_MOCK_DATA) {
    await new Promise(resolve => setTimeout(resolve, 80));
    return { ...MOCK_TIER1, method };
  }

  const backendId = ASSET_ID_MAP[assetId];
  if (!backendId) {
    throw new Error(`Unknown asset: ${assetId}`);
  }

  const response = await fetch(getApiUrl(`/ensemble/tier1/${backendId}?method=${method}`));
  if (!response.ok) {
    console.warn(`Tier 1 ensemble fetch failed for ${assetId}, using mock data`);
    return { ...MOCK_TIER1, method };
  }

  return response.json();
}

/**
 * Get Tier 2 ensemble prediction for an asset
 * Tier 2 methods: Bayesian model averaging, regime-adaptive, conformal prediction
 */
export async function getTier2Ensemble(
  assetId: AssetId,
  method: Tier2Method = "combined"
): Promise<Tier2Result> {
  if (USE_MOCK_DATA) {
    await new Promise(resolve => setTimeout(resolve, 100));
    return { ...MOCK_TIER2, method };
  }

  const backendId = ASSET_ID_MAP[assetId];
  if (!backendId) {
    throw new Error(`Unknown asset: ${assetId}`);
  }

  const response = await fetch(getApiUrl(`/ensemble/tier2/${backendId}?method=${method}`));
  if (!response.ok) {
    console.warn(`Tier 2 ensemble fetch failed for ${assetId}, using mock data`);
    return { ...MOCK_TIER2, method };
  }

  return response.json();
}

/**
 * Get Tier 3 ensemble prediction for an asset
 * Tier 3 methods: Thompson sampling, attention-based, quantile regression forest
 */
export async function getTier3Ensemble(
  assetId: AssetId,
  method: Tier3Method = "combined"
): Promise<Tier3Result> {
  if (USE_MOCK_DATA) {
    await new Promise(resolve => setTimeout(resolve, 150));
    return { ...MOCK_TIER3, method };
  }

  const backendId = ASSET_ID_MAP[assetId];
  if (!backendId) {
    throw new Error(`Unknown asset: ${assetId}`);
  }

  const response = await fetch(getApiUrl(`/ensemble/tier3/${backendId}?method=${method}`));
  if (!response.ok) {
    console.warn(`Tier 3 ensemble fetch failed for ${assetId}, using mock data`);
    return { ...MOCK_TIER3, method };
  }

  return response.json();
}

/**
 * Get all tier predictions for an asset (alias for getTierComparison for naming consistency)
 */
export async function getAllTiers(assetId: AssetId): Promise<TierComparisonData> {
  return getTierComparison(assetId);
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

// ============================================================================
// Walk-Forward Validation API
// ============================================================================

import type {
  WalkForwardMethod,
  WalkForwardResponse,
  FoldResult,
  SummaryMetrics,
  EquityPoint,
  CostComparison,
} from "@/types/backtest";

/**
 * Generate mock walk-forward validation results
 * In production, this would call the backend API
 */
export async function getWalkForwardResults(
  assetId: AssetId,
  methods: WalkForwardMethod[],
  nFolds: number = 5
): Promise<WalkForwardResponse> {
  // Simulate API delay
  await new Promise((r) => setTimeout(r, 800));

  const methodResults: Record<string, FoldResult[]> = {};
  const summaryMetrics: Record<string, SummaryMetrics> = {};
  const equityCurves: Record<string, EquityPoint[]> = {};

  const baseDate = new Date("2024-01-01");

  methods.forEach((method) => {
    const folds: FoldResult[] = [];
    const tierMultiplier = method.startsWith("tier3")
      ? 1.15
      : method.startsWith("tier2")
        ? 1.08
        : 1.0;

    for (let i = 0; i < nFolds; i++) {
      const baseAccuracy = 52 + Math.random() * 12 * tierMultiplier;
      const baseSharpe = 0.8 + Math.random() * 1.2 * tierMultiplier;
      const baseReturn = 5 + Math.random() * 15 * tierMultiplier;

      const trainStart = new Date(baseDate);
      trainStart.setDate(trainStart.getDate() + i * 60);
      const trainEnd = new Date(trainStart);
      trainEnd.setDate(trainEnd.getDate() + 42);
      const testStart = new Date(trainEnd);
      testStart.setDate(testStart.getDate() + 1);
      const testEnd = new Date(testStart);
      testEnd.setDate(testEnd.getDate() + 18);

      const totalCosts = 50 + Math.random() * 100;
      const costDrag = (totalCosts / 10000) * 100;

      folds.push({
        fold_id: i + 1,
        train_start: trainStart.toISOString().split("T")[0],
        train_end: trainEnd.toISOString().split("T")[0],
        test_start: testStart.toISOString().split("T")[0],
        test_end: testEnd.toISOString().split("T")[0],
        n_train: 42,
        n_test: 18,
        accuracy: baseAccuracy + (Math.random() - 0.5) * 8,
        sharpe_ratio: baseSharpe + (Math.random() - 0.5) * 0.4,
        sortino_ratio: baseSharpe * 1.2 + (Math.random() - 0.5) * 0.3,
        max_drawdown: -(8 + Math.random() * 12),
        win_rate: 48 + Math.random() * 15,
        total_return: baseReturn + (Math.random() - 0.5) * 6,
        n_trades: 15 + Math.floor(Math.random() * 20),
        avg_trade_return: 0.3 + Math.random() * 0.8,
        avg_holding_days: 3 + Math.random() * 4,
        n_bullish: 8 + Math.floor(Math.random() * 8),
        n_bearish: 6 + Math.floor(Math.random() * 6),
        n_neutral: 2 + Math.floor(Math.random() * 4),
        total_costs: totalCosts,
        avg_cost_per_trade: totalCosts / (15 + Math.random() * 10),
        avg_cost_bps: 4 + Math.random() * 3,
        cost_drag_pct: costDrag,
        regime_performance: {
          bull: {
            regime: "bull",
            n_samples: 8 + Math.floor(Math.random() * 5),
            accuracy: baseAccuracy + 5 + Math.random() * 5,
            sharpe_ratio: baseSharpe + 0.3,
            total_return: baseReturn * 1.3,
            win_rate: 55 + Math.random() * 10,
          },
          bear: {
            regime: "bear",
            n_samples: 5 + Math.floor(Math.random() * 4),
            accuracy: baseAccuracy - 3 + Math.random() * 4,
            sharpe_ratio: baseSharpe - 0.2,
            total_return: baseReturn * 0.7,
            win_rate: 45 + Math.random() * 10,
          },
          sideways: {
            regime: "sideways",
            n_samples: 4 + Math.floor(Math.random() * 3),
            accuracy: baseAccuracy + Math.random() * 3,
            sharpe_ratio: baseSharpe * 0.9,
            total_return: baseReturn * 0.5,
            win_rate: 50 + Math.random() * 8,
          },
        },
        returns: Array.from({ length: 18 }, () => (Math.random() - 0.48) * 2),
      });
    }

    methodResults[method] = folds;

    // Calculate summary metrics
    const avgAcc = folds.reduce((s, f) => s + f.accuracy, 0) / nFolds;
    const avgSharpe = folds.reduce((s, f) => s + f.sharpe_ratio, 0) / nFolds;
    const avgReturn = folds.reduce((s, f) => s + f.total_return, 0) / nFolds;
    const avgDrawdown = folds.reduce((s, f) => s + f.max_drawdown, 0) / nFolds;
    const avgWinRate = folds.reduce((s, f) => s + f.win_rate, 0) / nFolds;
    const avgCostDrag = folds.reduce((s, f) => s + f.cost_drag_pct, 0) / nFolds;
    const totalCosts = folds.reduce((s, f) => s + f.total_costs, 0);

    summaryMetrics[method] = {
      mean_accuracy: avgAcc,
      mean_sharpe: avgSharpe,
      mean_sortino: avgSharpe * 1.15,
      mean_max_drawdown: avgDrawdown,
      mean_win_rate: avgWinRate,
      mean_total_return: avgReturn,
      std_accuracy: 3 + Math.random() * 2,
      std_sharpe: 0.2 + Math.random() * 0.15,
      std_total_return: 2 + Math.random() * 2,
      mean_cost_drag_pct: avgCostDrag,
      total_costs: totalCosts,
      raw_total_return: avgReturn + avgCostDrag,
      cost_adjusted_return: avgReturn,
    };

    // Generate equity curve
    const curve: EquityPoint[] = [];
    let equity = 100000;
    let peak = equity;
    const days = 250;

    for (let d = 0; d < days; d++) {
      const date = new Date(baseDate);
      date.setDate(date.getDate() + d);
      const dailyReturn = (avgReturn / 252 + (Math.random() - 0.5) * 0.02) / 100;
      equity *= 1 + dailyReturn;
      if (equity > peak) peak = equity;
      const drawdown = (equity - peak) / peak;

      curve.push({
        date: date.toISOString().split("T")[0],
        equity: Math.round(equity),
        drawdown,
        benchmark: 100000 * (1 + (d / 252) * 0.08),
        returns: dailyReturn * 100,
      });
    }
    equityCurves[method] = curve;
  });

  // Generate rankings
  const rankings: Record<string, number> = {};
  const sortedMethods = [...methods].sort(
    (a, b) =>
      (summaryMetrics[b]?.mean_sharpe ?? 0) - (summaryMetrics[a]?.mean_sharpe ?? 0)
  );
  sortedMethods.forEach((m, i) => {
    rankings[m] = i + 1;
  });

  return {
    success: true,
    data: {
      asset_id: assetId,
      asset_name: assetId.replace("-", " ").replace(/\b\w/g, (c) => c.toUpperCase()),
      n_folds: nFolds,
      timestamp: new Date().toISOString(),
      method_results: methodResults as Record<WalkForwardMethod, FoldResult[]>,
      summary_metrics: summaryMetrics as Record<WalkForwardMethod, SummaryMetrics>,
      significance_tests: {},
      rankings: rankings as Record<WalkForwardMethod, number>,
    },
    equity_curves: equityCurves as Record<WalkForwardMethod, EquityPoint[]>,
  };
}

/**
 * Get cost comparison data for methods
 */
export function getCostComparison(
  summaryMetrics: Record<WalkForwardMethod, SummaryMetrics>
): CostComparison[] {
  return Object.entries(summaryMetrics).map(([method, metrics]) => ({
    method: method as WalkForwardMethod,
    raw_return: metrics.raw_total_return,
    cost_adjusted_return: metrics.cost_adjusted_return,
    cost_impact: metrics.raw_total_return - metrics.cost_adjusted_return,
    raw_sharpe: metrics.mean_sharpe + 0.1,
    cost_adjusted_sharpe: metrics.mean_sharpe,
  }));
}
