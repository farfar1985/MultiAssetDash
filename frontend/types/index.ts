// Persona types for QDT Nexus
export type PersonaId =
  | "quant"
  | "procurement"
  | "hedging"
  | "hedgefund"
  | "alphapro"
  | "proretail"
  | "retail"
  | "executive";

export interface Persona {
  id: PersonaId;
  name: string;
  description: string;
  icon: string;
}

export const PERSONAS: Record<PersonaId, Persona> = {
  quant: {
    id: "quant",
    name: "Hardcore Quant",
    description: "Advanced quantitative analysis and model validation",
    icon: "chart-bar",
  },
  procurement: {
    id: "procurement",
    name: "Procurement Team",
    description: "Strategic sourcing and supply chain optimization",
    icon: "truck",
  },
  hedging: {
    id: "hedging",
    name: "Hedging Team",
    description: "Risk management and hedge execution",
    icon: "shield-check",
  },
  hedgefund: {
    id: "hedgefund",
    name: "Hedge Fund",
    description: "Alpha generation and portfolio optimization",
    icon: "trending-up",
  },
  alphapro: {
    id: "alphapro",
    name: "Alpha Gen Pro",
    description: "Professional alpha signal discovery",
    icon: "zap",
  },
  proretail: {
    id: "proretail",
    name: "Pro Retail",
    description: "Advanced retail trading insights",
    icon: "user-check",
  },
  retail: {
    id: "retail",
    name: "Retail Trader",
    description: "Simple buy/sell/hold signals in plain English",
    icon: "sparkles",
  },
  executive: {
    id: "executive",
    name: "Executive",
    description: "Headline metrics and market outlook for leadership",
    icon: "briefcase",
  },
};

// Asset types
export type AssetId =
  | "crude-oil"
  | "bitcoin"
  | "gold"
  | "silver"
  | "natural-gas"
  | "copper"
  | "wheat"
  | "corn"
  | "soybean"
  | "platinum";

export interface Asset {
  id: AssetId;
  name: string;
  symbol: string;
  category: "energy" | "metals" | "crypto" | "agriculture";
}

export const ASSETS: Record<AssetId, Asset> = {
  "crude-oil": {
    id: "crude-oil",
    name: "Crude Oil",
    symbol: "CL",
    category: "energy",
  },
  bitcoin: {
    id: "bitcoin",
    name: "Bitcoin",
    symbol: "BTC",
    category: "crypto",
  },
  gold: {
    id: "gold",
    name: "Gold",
    symbol: "GC",
    category: "metals",
  },
  silver: {
    id: "silver",
    name: "Silver",
    symbol: "SI",
    category: "metals",
  },
  "natural-gas": {
    id: "natural-gas",
    name: "Natural Gas",
    symbol: "NG",
    category: "energy",
  },
  copper: {
    id: "copper",
    name: "Copper",
    symbol: "HG",
    category: "metals",
  },
  wheat: {
    id: "wheat",
    name: "Wheat",
    symbol: "ZW",
    category: "agriculture",
  },
  corn: {
    id: "corn",
    name: "Corn",
    symbol: "ZC",
    category: "agriculture",
  },
  soybean: {
    id: "soybean",
    name: "Soybean",
    symbol: "ZS",
    category: "agriculture",
  },
  platinum: {
    id: "platinum",
    name: "Platinum",
    symbol: "PL",
    category: "metals",
  },
};

// Signal types
export type SignalDirection = "bullish" | "bearish" | "neutral";
export type SignalStrength = "strong" | "moderate" | "weak";
export type Timeframe = "1h" | "4h" | "1d" | "1w" | "1m";

export interface Signal {
  id: string;
  assetId: AssetId;
  direction: SignalDirection;
  strength: SignalStrength;
  confidence: number; // 0-100
  timeframe: Timeframe;
  entryPrice?: number;
  targetPrice?: number;
  stopLoss?: number;
  generatedAt: string;
  expiresAt?: string;
  rationale?: string;
}

// Metrics types
export interface MetricValue {
  value: number;
  change: number;
  changePercent: number;
  trend: "up" | "down" | "flat";
}

export interface DashboardMetrics {
  portfolioValue: MetricValue;
  dailyPnL: MetricValue;
  winRate: MetricValue;
  sharpeRatio: MetricValue;
  activeSignals: number;
  openPositions: number;
}

// API response types
export interface ApiResponse<T> {
  data: T;
  success: boolean;
  error?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    pageSize: number;
    total: number;
    totalPages: number;
  };
}

// ============================================================================
// Quantum ML Types
// ============================================================================

/**
 * Official backtest result metrics from quantum_ml pipeline
 * These are the canonical performance metrics for strategy evaluation
 */
export interface BacktestResult {
  // Risk-adjusted returns
  sharpe: number;
  sortino: number;
  information_ratio: number;

  // Drawdown metrics
  max_drawdown: number;
  avg_drawdown: number;
  max_drawdown_duration_days: number;

  // ROI at different periods (percentage)
  roi_30: number;
  roi_60: number;
  roi_90: number;
  roi_180: number;
  roi_360: number;

  // Win/loss metrics
  win_rate: number;
  profit_factor: number;
  avg_win: number;
  avg_loss: number;
  win_loss_ratio: number;

  // Trade statistics
  total_trades: number;
  avg_trade_duration_days: number;

  // Volatility
  annualized_volatility: number;
  downside_volatility: number;

  // Additional context
  strategy: string;
  symbol: string;
  start_date: string;
  end_date: string;
}

/**
 * Model accuracy metrics from quantum_ml
 * Tracks directional prediction accuracy
 */
export interface ModelAccuracy {
  model_id: string;
  model_name: string;

  // Overall accuracy
  acc_predict: number;

  // Directional accuracy breakdown
  acc_predict_up: number;
  acc_predict_down: number;
  acc_predict_abs: number;

  // Confidence metrics
  avg_confidence: number;
  calibration_error: number;

  // Sample info
  sample_size: number;
  evaluation_period_start: string;
  evaluation_period_end: string;
}

/**
 * Feature importance from quantum_ml models
 * Maps feature names to their importance scores
 */
export interface FeatureImportance {
  feature: string;
  importance: number;
  category: "technical" | "fundamental" | "sentiment" | "macro" | "derived";
  description?: string;
}

/**
 * Historical metrics for time-travel functionality
 * Allows viewing backtest results as of a specific historical date
 */
export interface HistoricalMetrics {
  as_of_date: string;
  metrics: BacktestResult;
  model_accuracies: ModelAccuracy[];
  snapshot_type: "daily" | "weekly" | "monthly";
}

/**
 * Strategy type identifier
 */
export type StrategyId =
  | "momentum"
  | "mean_reversion"
  | "trend_following"
  | "breakout"
  | "ensemble_weighted"
  | "ml_classifier"
  | "hybrid";

/**
 * Time range for historical queries
 */
export interface TimeRange {
  start_date: string;
  end_date: string;
}

// Re-export backtest types
export * from "./backtest";
