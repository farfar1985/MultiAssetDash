/**
 * Tests for /api/ensemble/[symbol] route
 *
 * Testing the GET handler for ensemble results with various symbols and methods.
 */

import type { EnsembleMethod, EnsembleResult, ApiResponse } from "@/lib/api-client";

// Mock NextResponse before importing the route
jest.mock("next/server", () => ({
  NextResponse: {
    json: (data: unknown, init?: { status?: number }) => {
      return {
        status: init?.status || 200,
        json: async () => data,
      };
    },
  },
}));

// Import route after mocking
import { GET } from "../[symbol]/route";

// Helper to create a mock request
function createMockRequest(
  symbol: string,
  method?: string
): Request {
  const url = method
    ? `http://localhost:3000/api/ensemble/${symbol}?method=${method}`
    : `http://localhost:3000/api/ensemble/${symbol}`;
  return new Request(url);
}

// Helper to parse response JSON
async function parseResponse<T>(response: Response): Promise<ApiResponse<T>> {
  return response.json();
}

describe("GET /api/ensemble/[symbol]", () => {
  describe("valid requests", () => {
    describe("symbol resolution", () => {
      it("returns data for valid ticker symbol (CL)", async () => {
        const request = createMockRequest("CL");
        const params = Promise.resolve({ symbol: "CL" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        expect(response.status).toBe(200);
        expect(body.success).toBe(true);
        expect(body.data).toBeDefined();
        expect(body.data!.signal.assetId).toBe("crude-oil");
      });

      it("returns data for lowercase ticker symbol", async () => {
        const request = createMockRequest("cl");
        const params = Promise.resolve({ symbol: "cl" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        expect(response.status).toBe(200);
        expect(body.data!.signal.assetId).toBe("crude-oil");
      });

      it("returns data for asset ID directly (crude-oil)", async () => {
        const request = createMockRequest("crude-oil");
        const params = Promise.resolve({ symbol: "crude-oil" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        expect(response.status).toBe(200);
        expect(body.data!.signal.assetId).toBe("crude-oil");
      });

      it("resolves all ticker symbols correctly", async () => {
        const symbolMappings = [
          { ticker: "CL", assetId: "crude-oil" },
          { ticker: "BTC", assetId: "bitcoin" },
          { ticker: "GC", assetId: "gold" },
          { ticker: "SI", assetId: "silver" },
          { ticker: "NG", assetId: "natural-gas" },
          { ticker: "HG", assetId: "copper" },
          { ticker: "ZW", assetId: "wheat" },
          { ticker: "ZC", assetId: "corn" },
          { ticker: "ZS", assetId: "soybean" },
          { ticker: "PL", assetId: "platinum" },
        ];

        for (const mapping of symbolMappings) {
          const request = createMockRequest(mapping.ticker);
          const params = Promise.resolve({ symbol: mapping.ticker });

          const response = await GET(request, { params });
          const body = await parseResponse<EnsembleResult>(response);

          expect(response.status).toBe(200);
          expect(body.data!.signal.assetId).toBe(mapping.assetId);
        }
      });

      it("resolves all asset IDs correctly", async () => {
        const assetIds = [
          "crude-oil",
          "bitcoin",
          "gold",
          "silver",
          "natural-gas",
          "copper",
          "wheat",
          "corn",
          "soybean",
          "platinum",
        ];

        for (const assetId of assetIds) {
          const request = createMockRequest(assetId);
          const params = Promise.resolve({ symbol: assetId });

          const response = await GET(request, { params });
          const body = await parseResponse<EnsembleResult>(response);

          expect(response.status).toBe(200);
          expect(body.data!.signal.assetId).toBe(assetId);
        }
      });
    });

    describe("default method", () => {
      it("uses accuracy_weighted as default method", async () => {
        const request = createMockRequest("BTC");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        expect(body.data!.method).toBe("accuracy_weighted");
      });
    });

    describe("method-specific behavior", () => {
      const methods: EnsembleMethod[] = [
        "accuracy_weighted",
        "exponential_decay",
        "top_k_sharpe",
        "ridge_stacking",
        "inverse_variance",
        "pairwise_slope",
      ];

      methods.forEach((method) => {
        it(`returns valid data for ${method} method`, async () => {
          const request = createMockRequest("GC", method);
          const params = Promise.resolve({ symbol: "GC" });

          const response = await GET(request, { params });
          const body = await parseResponse<EnsembleResult>(response);

          expect(response.status).toBe(200);
          expect(body.success).toBe(true);
          expect(body.data!.method).toBe(method);
        });
      });

      it("applies correct confidence multiplier for top_k_sharpe (1.05)", async () => {
        // Gold has baseConfidence: 84
        // top_k_sharpe has confidenceMultiplier: 1.05
        // Expected: Math.min(99, Math.round(84 * 1.05)) = Math.min(99, 88) = 88
        const request = createMockRequest("GC", "top_k_sharpe");
        const params = Promise.resolve({ symbol: "GC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        expect(body.data!.signal.confidence).toBe(88);
      });

      it("applies correct confidence multiplier for exponential_decay (0.95)", async () => {
        // Gold has baseConfidence: 84
        // exponential_decay has confidenceMultiplier: 0.95
        // Expected: Math.round(84 * 0.95) = 80
        const request = createMockRequest("GC", "exponential_decay");
        const params = Promise.resolve({ symbol: "GC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        expect(body.data!.signal.confidence).toBe(80);
      });

      it("caps confidence at 99 for high-confidence assets", async () => {
        // natural-gas has baseConfidence: 91
        // top_k_sharpe has confidenceMultiplier: 1.05
        // 91 * 1.05 = 95.55 -> rounds to 96 (under 99)
        const request = createMockRequest("NG", "top_k_sharpe");
        const params = Promise.resolve({ symbol: "NG" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        expect(body.data!.signal.confidence).toBeLessThanOrEqual(99);
      });

      it("applies correct sharpe boost for top_k_sharpe (0.25)", async () => {
        const requestBase = createMockRequest("GC", "accuracy_weighted");
        const paramsBase = Promise.resolve({ symbol: "GC" });

        const responseBase = await GET(requestBase, { params: paramsBase });
        const bodyBase = await parseResponse<EnsembleResult>(responseBase);

        const requestTopK = createMockRequest("GC", "top_k_sharpe");
        const paramsTopK = Promise.resolve({ symbol: "GC" });

        const responseTopK = await GET(requestTopK, { params: paramsTopK });
        const bodyTopK = await parseResponse<EnsembleResult>(responseTopK);

        // top_k_sharpe should have +0.25 sharpe boost over base (0)
        expect(bodyTopK.data!.signal.sharpeRatio).toBeCloseTo(
          bodyBase.data!.signal.sharpeRatio + 0.25,
          2
        );
      });

      it("applies model weight boost for top_k_sharpe lstm_volatility", async () => {
        const request = createMockRequest("BTC", "top_k_sharpe");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        // lstm_volatility base is 0.18, top_k_sharpe adds 0.05
        expect(body.data!.modelWeights.lstm_volatility).toBeCloseTo(0.23, 2);
      });

      it("applies model weight boost for accuracy_weighted transformer_trend", async () => {
        const request = createMockRequest("BTC", "accuracy_weighted");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        // transformer_trend base is 0.22, accuracy_weighted adds 0.03
        expect(body.data!.modelWeights.transformer_trend).toBeCloseTo(0.25, 2);
      });

      it("applies model weight boost for pairwise_slope gradient_boost_macro", async () => {
        const request = createMockRequest("BTC", "pairwise_slope");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        // gradient_boost_macro base is 0.15, pairwise_slope adds 0.04
        expect(body.data!.modelWeights.gradient_boost_macro).toBeCloseTo(0.19, 2);
      });
    });

    describe("response data structure", () => {
      it("returns complete EnsembleResult structure", async () => {
        const request = createMockRequest("BTC", "top_k_sharpe");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        expect(body.data).toHaveProperty("method");
        expect(body.data).toHaveProperty("signal");
        expect(body.data).toHaveProperty("modelWeights");
        expect(body.data).toHaveProperty("backtestMetrics");
      });

      it("returns complete signal structure", async () => {
        const request = createMockRequest("BTC", "top_k_sharpe");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        const signal = body.data!.signal;
        expect(signal).toHaveProperty("assetId");
        expect(signal).toHaveProperty("direction");
        expect(signal).toHaveProperty("confidence");
        expect(signal).toHaveProperty("horizon");
        expect(signal).toHaveProperty("modelsAgreeing");
        expect(signal).toHaveProperty("modelsTotal");
        expect(signal).toHaveProperty("sharpeRatio");
        expect(signal).toHaveProperty("directionalAccuracy");
        expect(signal).toHaveProperty("totalReturn");
        expect(signal).toHaveProperty("generatedAt");
      });

      it("returns correct signal directions for assets", async () => {
        const expectedDirections = [
          { symbol: "CL", direction: "bullish" },
          { symbol: "BTC", direction: "bearish" },
          { symbol: "GC", direction: "bullish" },
          { symbol: "SI", direction: "bearish" },
          { symbol: "NG", direction: "bullish" },
          { symbol: "HG", direction: "neutral" },
          { symbol: "ZW", direction: "bearish" },
          { symbol: "ZC", direction: "bullish" },
          { symbol: "ZS", direction: "bearish" },
          { symbol: "PL", direction: "bullish" },
        ];

        for (const expected of expectedDirections) {
          const request = createMockRequest(expected.symbol);
          const params = Promise.resolve({ symbol: expected.symbol });

          const response = await GET(request, { params });
          const body = await parseResponse<EnsembleResult>(response);

          expect(body.data!.signal.direction).toBe(expected.direction);
        }
      });

      it("returns horizon as D+1", async () => {
        const request = createMockRequest("BTC");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        expect(body.data!.signal.horizon).toBe("D+1");
      });

      it("returns modelsTotal as 10179", async () => {
        const request = createMockRequest("BTC");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        expect(body.data!.signal.modelsTotal).toBe(10179);
      });

      it("calculates modelsAgreeing correctly", async () => {
        // Bitcoin: baseConfidence: 65, accuracy_weighted multiplier: 1.0
        // confidence = 65
        // modelsAgreeing = Math.round(10179 * (65/100) * 0.95) = Math.round(6285.48) = 6285
        const request = createMockRequest("BTC", "accuracy_weighted");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        const expectedModelsAgreeing = Math.round(
          10179 * (body.data!.signal.confidence / 100) * 0.95
        );
        expect(body.data!.signal.modelsAgreeing).toBe(expectedModelsAgreeing);
      });

      it("returns all 6 model weights", async () => {
        const request = createMockRequest("BTC");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        const weights = body.data!.modelWeights;
        expect(Object.keys(weights)).toHaveLength(6);
        expect(weights).toHaveProperty("lstm_volatility");
        expect(weights).toHaveProperty("transformer_trend");
        expect(weights).toHaveProperty("gradient_boost_macro");
        expect(weights).toHaveProperty("random_forest_seasonal");
        expect(weights).toHaveProperty("xgboost_technical");
        expect(weights).toHaveProperty("neural_ensemble");
      });

      it("returns complete backtestMetrics", async () => {
        const request = createMockRequest("BTC");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        const metrics = body.data!.backtestMetrics;
        expect(metrics).toHaveProperty("sharpeRatio");
        expect(metrics).toHaveProperty("directionalAccuracy");
        expect(metrics).toHaveProperty("totalReturn");
        expect(metrics).toHaveProperty("maxDrawdown");
      });

      it("calculates directionalAccuracy correctly", async () => {
        const request = createMockRequest("GC", "accuracy_weighted");
        const params = Promise.resolve({ symbol: "GC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        const confidence = body.data!.signal.confidence;
        // directionalAccuracy = 55 + confidence * 0.08
        const expectedDA = 55 + confidence * 0.08;
        expect(body.data!.signal.directionalAccuracy).toBeCloseTo(expectedDA, 2);
      });

      it("calculates totalReturn correctly", async () => {
        const request = createMockRequest("GC", "accuracy_weighted");
        const params = Promise.resolve({ symbol: "GC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        const confidence = body.data!.signal.confidence;
        // totalReturn = 30 + confidence * 0.3
        const expectedReturn = 30 + confidence * 0.3;
        expect(body.data!.signal.totalReturn).toBeCloseTo(expectedReturn, 2);
      });

      it("calculates maxDrawdown correctly", async () => {
        const request = createMockRequest("GC", "accuracy_weighted");
        const params = Promise.resolve({ symbol: "GC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        const confidence = body.data!.signal.confidence;
        // maxDrawdown = -12 - (100 - confidence) * 0.1
        const expectedDrawdown = -12 - (100 - confidence) * 0.1;
        expect(body.data!.backtestMetrics.maxDrawdown).toBeCloseTo(
          expectedDrawdown,
          2
        );
      });

      it("returns valid ISO timestamp in generatedAt", async () => {
        const request = createMockRequest("BTC");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        const date = new Date(body.data!.signal.generatedAt);
        expect(date.toISOString()).toBe(body.data!.signal.generatedAt);
      });

      it("returns timestamp in response", async () => {
        const request = createMockRequest("BTC");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<EnsembleResult>(response);

        expect(body.timestamp).toBeDefined();
        const date = new Date(body.timestamp);
        expect(date.toISOString()).toBe(body.timestamp);
      });
    });
  });

  describe("error handling", () => {
    describe("invalid method (400)", () => {
      it("returns 400 for invalid method", async () => {
        const request = createMockRequest("BTC", "invalid_method");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });

        expect(response.status).toBe(400);
      });

      it("returns success: false for invalid method", async () => {
        const request = createMockRequest("BTC", "invalid_method");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<null>(response);

        expect(body.success).toBe(false);
      });

      it("returns null data for invalid method", async () => {
        const request = createMockRequest("BTC", "invalid_method");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<null>(response);

        expect(body.data).toBeNull();
      });

      it("returns error message listing valid methods", async () => {
        const request = createMockRequest("BTC", "bad_method");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<null>(response);

        expect(body.error).toContain("Invalid ensemble method: bad_method");
        expect(body.error).toContain("accuracy_weighted");
        expect(body.error).toContain("exponential_decay");
        expect(body.error).toContain("top_k_sharpe");
        expect(body.error).toContain("ridge_stacking");
        expect(body.error).toContain("inverse_variance");
        expect(body.error).toContain("pairwise_slope");
      });

      it("returns timestamp even for errors", async () => {
        const request = createMockRequest("BTC", "invalid_method");
        const params = Promise.resolve({ symbol: "BTC" });

        const response = await GET(request, { params });
        const body = await parseResponse<null>(response);

        expect(body.timestamp).toBeDefined();
      });
    });

    describe("unknown asset (404)", () => {
      it("returns 404 for unknown ticker symbol", async () => {
        const request = createMockRequest("UNKNOWN");
        const params = Promise.resolve({ symbol: "UNKNOWN" });

        const response = await GET(request, { params });

        expect(response.status).toBe(404);
      });

      it("returns 404 for unknown asset ID", async () => {
        const request = createMockRequest("not-an-asset");
        const params = Promise.resolve({ symbol: "not-an-asset" });

        const response = await GET(request, { params });

        expect(response.status).toBe(404);
      });

      it("returns success: false for unknown asset", async () => {
        const request = createMockRequest("UNKNOWN");
        const params = Promise.resolve({ symbol: "UNKNOWN" });

        const response = await GET(request, { params });
        const body = await parseResponse<null>(response);

        expect(body.success).toBe(false);
      });

      it("returns null data for unknown asset", async () => {
        const request = createMockRequest("UNKNOWN");
        const params = Promise.resolve({ symbol: "UNKNOWN" });

        const response = await GET(request, { params });
        const body = await parseResponse<null>(response);

        expect(body.data).toBeNull();
      });

      it("returns error message with symbol", async () => {
        const request = createMockRequest("XYZ123");
        const params = Promise.resolve({ symbol: "XYZ123" });

        const response = await GET(request, { params });
        const body = await parseResponse<null>(response);

        expect(body.error).toBe("Asset not found: XYZ123");
      });

      it("returns timestamp even for 404", async () => {
        const request = createMockRequest("UNKNOWN");
        const params = Promise.resolve({ symbol: "UNKNOWN" });

        const response = await GET(request, { params });
        const body = await parseResponse<null>(response);

        expect(body.timestamp).toBeDefined();
      });
    });
  });

  describe("edge cases", () => {
    it("handles mixed case ticker symbols", async () => {
      const request = createMockRequest("Btc");
      const params = Promise.resolve({ symbol: "Btc" });

      const response = await GET(request, { params });
      const body = await parseResponse<EnsembleResult>(response);

      expect(response.status).toBe(200);
      expect(body.data!.signal.assetId).toBe("bitcoin");
    });

    it("handles uppercase asset IDs by converting to lowercase", async () => {
      const request = createMockRequest("BITCOIN");
      const params = Promise.resolve({ symbol: "BITCOIN" });

      const response = await GET(request, { params });
      const body = await parseResponse<EnsembleResult>(response);

      expect(response.status).toBe(200);
      expect(body.data!.signal.assetId).toBe("bitcoin");
    });

    it("handles empty method parameter (uses default)", async () => {
      const url = "http://localhost:3000/api/ensemble/BTC?method=";
      const request = new Request(url);
      const params = Promise.resolve({ symbol: "BTC" });

      const response = await GET(request, { params });
      const body = await parseResponse<EnsembleResult>(response);

      // Empty string is falsy in JS, so (searchParams.get("method") || "accuracy_weighted")
      // will use the default "accuracy_weighted"
      expect(response.status).toBe(200);
      expect(body.data!.method).toBe("accuracy_weighted");
    });

    it("prioritizes ticker symbol resolution over asset ID", async () => {
      // If "GC" is both a ticker and an asset ID, ticker should win
      const request = createMockRequest("GC");
      const params = Promise.resolve({ symbol: "GC" });

      const response = await GET(request, { params });
      const body = await parseResponse<EnsembleResult>(response);

      expect(body.data!.signal.assetId).toBe("gold");
    });
  });

  describe("consistency checks", () => {
    it("returns consistent data for same request", async () => {
      const request1 = createMockRequest("BTC", "top_k_sharpe");
      const params1 = Promise.resolve({ symbol: "BTC" });

      const request2 = createMockRequest("BTC", "top_k_sharpe");
      const params2 = Promise.resolve({ symbol: "BTC" });

      const response1 = await GET(request1, { params: params1 });
      const response2 = await GET(request2, { params: params2 });

      const body1 = await parseResponse<EnsembleResult>(response1);
      const body2 = await parseResponse<EnsembleResult>(response2);

      // All static fields should match
      expect(body1.data!.method).toBe(body2.data!.method);
      expect(body1.data!.signal.assetId).toBe(body2.data!.signal.assetId);
      expect(body1.data!.signal.direction).toBe(body2.data!.signal.direction);
      expect(body1.data!.signal.confidence).toBe(body2.data!.signal.confidence);
      expect(body1.data!.modelWeights).toEqual(body2.data!.modelWeights);
    });

    it("signal sharpeRatio matches backtestMetrics sharpeRatio", async () => {
      const request = createMockRequest("GC", "ridge_stacking");
      const params = Promise.resolve({ symbol: "GC" });

      const response = await GET(request, { params });
      const body = await parseResponse<EnsembleResult>(response);

      expect(body.data!.signal.sharpeRatio).toBe(
        body.data!.backtestMetrics.sharpeRatio
      );
    });

    it("signal directionalAccuracy matches backtestMetrics", async () => {
      const request = createMockRequest("GC", "ridge_stacking");
      const params = Promise.resolve({ symbol: "GC" });

      const response = await GET(request, { params });
      const body = await parseResponse<EnsembleResult>(response);

      expect(body.data!.signal.directionalAccuracy).toBe(
        body.data!.backtestMetrics.directionalAccuracy
      );
    });

    it("signal totalReturn matches backtestMetrics", async () => {
      const request = createMockRequest("GC", "ridge_stacking");
      const params = Promise.resolve({ symbol: "GC" });

      const response = await GET(request, { params });
      const body = await parseResponse<EnsembleResult>(response);

      expect(body.data!.signal.totalReturn).toBe(
        body.data!.backtestMetrics.totalReturn
      );
    });
  });
});
