import { renderHook, act } from "@testing-library/react";
import { ReactNode } from "react";
import {
  EnsembleProvider,
  useEnsembleContext,
  useEnsembleMethod,
} from "../EnsembleContext";
import type { EnsembleMethod, EnsembleResult } from "@/lib/api-client";

// Helper to create a wrapper for hooks
function createWrapper(defaultMethod?: EnsembleMethod) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <EnsembleProvider defaultMethod={defaultMethod}>
        {children}
      </EnsembleProvider>
    );
  };
}

// Mock EnsembleResult for testing cache functionality
function createMockResult(
  method: EnsembleMethod,
  assetId: string
): EnsembleResult {
  return {
    method,
    signal: {
      assetId: assetId as any,
      direction: "bullish",
      confidence: 75,
      horizon: "D+1",
      modelsAgreeing: 7500,
      modelsTotal: 10000,
      sharpeRatio: 2.5,
      directionalAccuracy: 60,
      totalReturn: 45,
      generatedAt: new Date().toISOString(),
    },
    modelWeights: {
      lstm_volatility: 0.2,
      transformer_trend: 0.25,
      gradient_boost_macro: 0.15,
      random_forest_seasonal: 0.15,
      xgboost_technical: 0.15,
      neural_ensemble: 0.1,
    },
    backtestMetrics: {
      sharpeRatio: 2.5,
      directionalAccuracy: 60,
      totalReturn: 45,
      maxDrawdown: -15,
    },
  };
}

describe("EnsembleContext", () => {
  describe("EnsembleProvider", () => {
    describe("default method initialization", () => {
      it("sets default method to top_k_sharpe when no defaultMethod provided", () => {
        const { result } = renderHook(() => useEnsembleContext(), {
          wrapper: createWrapper(),
        });

        expect(result.current.currentMethod).toBe("top_k_sharpe");
      });

      it("sets custom default method when provided", () => {
        const { result } = renderHook(() => useEnsembleContext(), {
          wrapper: createWrapper("accuracy_weighted"),
        });

        expect(result.current.currentMethod).toBe("accuracy_weighted");
      });

      it("accepts all valid ensemble methods as default", () => {
        const methods: EnsembleMethod[] = [
          "accuracy_weighted",
          "exponential_decay",
          "top_k_sharpe",
          "ridge_stacking",
          "inverse_variance",
          "pairwise_slope",
        ];

        methods.forEach((method) => {
          const { result } = renderHook(() => useEnsembleContext(), {
            wrapper: createWrapper(method),
          });
          expect(result.current.currentMethod).toBe(method);
        });
      });

      it("initializes methodResults as empty object", () => {
        const { result } = renderHook(() => useEnsembleContext(), {
          wrapper: createWrapper(),
        });

        expect(result.current.methodResults).toEqual({});
      });
    });
  });

  describe("setMethod", () => {
    it("updates current method when called", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      expect(result.current.currentMethod).toBe("top_k_sharpe");

      act(() => {
        result.current.setMethod("accuracy_weighted");
      });

      expect(result.current.currentMethod).toBe("accuracy_weighted");
    });

    it("can change method multiple times", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      act(() => {
        result.current.setMethod("accuracy_weighted");
      });
      expect(result.current.currentMethod).toBe("accuracy_weighted");

      act(() => {
        result.current.setMethod("exponential_decay");
      });
      expect(result.current.currentMethod).toBe("exponential_decay");

      act(() => {
        result.current.setMethod("ridge_stacking");
      });
      expect(result.current.currentMethod).toBe("ridge_stacking");
    });

    it("allows setting same method (no-op but valid)", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper("inverse_variance"),
      });

      act(() => {
        result.current.setMethod("inverse_variance");
      });

      expect(result.current.currentMethod).toBe("inverse_variance");
    });

    it("maintains callback stability across renders", () => {
      const { result, rerender } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const setMethodRef1 = result.current.setMethod;
      rerender();
      const setMethodRef2 = result.current.setMethod;

      expect(setMethodRef1).toBe(setMethodRef2);
    });
  });

  describe("cacheResult", () => {
    it("stores result for a symbol and method", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const mockResult = createMockResult("accuracy_weighted", "crude-oil");

      act(() => {
        result.current.cacheResult("crude-oil", "accuracy_weighted", mockResult);
      });

      expect(result.current.methodResults).toEqual({
        "crude-oil": {
          accuracy_weighted: mockResult,
        },
      });
    });

    it("stores multiple methods for same symbol", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const mockResult1 = createMockResult("accuracy_weighted", "bitcoin");
      const mockResult2 = createMockResult("top_k_sharpe", "bitcoin");

      act(() => {
        result.current.cacheResult("bitcoin", "accuracy_weighted", mockResult1);
        result.current.cacheResult("bitcoin", "top_k_sharpe", mockResult2);
      });

      expect(result.current.methodResults["bitcoin"]).toEqual({
        accuracy_weighted: mockResult1,
        top_k_sharpe: mockResult2,
      });
    });

    it("stores results for multiple symbols", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const goldResult = createMockResult("inverse_variance", "gold");
      const silverResult = createMockResult("inverse_variance", "silver");

      act(() => {
        result.current.cacheResult("gold", "inverse_variance", goldResult);
        result.current.cacheResult("silver", "inverse_variance", silverResult);
      });

      expect(result.current.methodResults["gold"]).toEqual({
        inverse_variance: goldResult,
      });
      expect(result.current.methodResults["silver"]).toEqual({
        inverse_variance: silverResult,
      });
    });

    it("overwrites existing cached result for same symbol and method", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const oldResult = createMockResult("ridge_stacking", "copper");
      const newResult = createMockResult("ridge_stacking", "copper");
      newResult.signal.confidence = 99;

      act(() => {
        result.current.cacheResult("copper", "ridge_stacking", oldResult);
      });

      expect(
        result.current.methodResults["copper"]?.["ridge_stacking"]?.signal.confidence
      ).toBe(75);

      act(() => {
        result.current.cacheResult("copper", "ridge_stacking", newResult);
      });

      expect(
        result.current.methodResults["copper"]?.["ridge_stacking"]?.signal.confidence
      ).toBe(99);
    });

    it("preserves existing symbol data when adding new method", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const result1 = createMockResult("accuracy_weighted", "wheat");
      const result2 = createMockResult("exponential_decay", "wheat");

      act(() => {
        result.current.cacheResult("wheat", "accuracy_weighted", result1);
      });

      act(() => {
        result.current.cacheResult("wheat", "exponential_decay", result2);
      });

      // Original result should still be present
      expect(result.current.methodResults["wheat"]?.["accuracy_weighted"]).toEqual(
        result1
      );
      expect(result.current.methodResults["wheat"]?.["exponential_decay"]).toEqual(
        result2
      );
    });

    it("maintains callback stability across renders", () => {
      const { result, rerender } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const cacheResultRef1 = result.current.cacheResult;
      rerender();
      const cacheResultRef2 = result.current.cacheResult;

      expect(cacheResultRef1).toBe(cacheResultRef2);
    });
  });

  describe("getCachedResult", () => {
    it("returns undefined when no cache exists for symbol", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const cached = result.current.getCachedResult(
        "unknown-symbol",
        "accuracy_weighted"
      );

      expect(cached).toBeUndefined();
    });

    it("returns undefined when symbol exists but method does not", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const mockResult = createMockResult("accuracy_weighted", "corn");

      act(() => {
        result.current.cacheResult("corn", "accuracy_weighted", mockResult);
      });

      const cached = result.current.getCachedResult("corn", "ridge_stacking");
      expect(cached).toBeUndefined();
    });

    it("returns cached result when it exists", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const mockResult = createMockResult("pairwise_slope", "soybean");

      act(() => {
        result.current.cacheResult("soybean", "pairwise_slope", mockResult);
      });

      const cached = result.current.getCachedResult("soybean", "pairwise_slope");
      expect(cached).toEqual(mockResult);
    });

    it("returns correct result when multiple symbols and methods cached", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const goldAccuracy = createMockResult("accuracy_weighted", "gold");
      const goldTopK = createMockResult("top_k_sharpe", "gold");
      const btcTopK = createMockResult("top_k_sharpe", "bitcoin");

      act(() => {
        result.current.cacheResult("gold", "accuracy_weighted", goldAccuracy);
        result.current.cacheResult("gold", "top_k_sharpe", goldTopK);
        result.current.cacheResult("bitcoin", "top_k_sharpe", btcTopK);
      });

      expect(
        result.current.getCachedResult("gold", "accuracy_weighted")
      ).toEqual(goldAccuracy);
      expect(result.current.getCachedResult("gold", "top_k_sharpe")).toEqual(
        goldTopK
      );
      expect(result.current.getCachedResult("bitcoin", "top_k_sharpe")).toEqual(
        btcTopK
      );
      expect(
        result.current.getCachedResult("bitcoin", "accuracy_weighted")
      ).toBeUndefined();
    });

    it("updates correctly when dependencies change", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      // Initially undefined
      expect(
        result.current.getCachedResult("platinum", "inverse_variance")
      ).toBeUndefined();

      const mockResult = createMockResult("inverse_variance", "platinum");

      act(() => {
        result.current.cacheResult("platinum", "inverse_variance", mockResult);
      });

      // Now should return the cached value
      expect(
        result.current.getCachedResult("platinum", "inverse_variance")
      ).toEqual(mockResult);
    });
  });

  describe("clearCache", () => {
    it("clears all cached results", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const result1 = createMockResult("accuracy_weighted", "crude-oil");
      const result2 = createMockResult("top_k_sharpe", "bitcoin");
      const result3 = createMockResult("ridge_stacking", "gold");

      act(() => {
        result.current.cacheResult("crude-oil", "accuracy_weighted", result1);
        result.current.cacheResult("bitcoin", "top_k_sharpe", result2);
        result.current.cacheResult("gold", "ridge_stacking", result3);
      });

      expect(Object.keys(result.current.methodResults).length).toBe(3);

      act(() => {
        result.current.clearCache();
      });

      expect(result.current.methodResults).toEqual({});
      expect(
        result.current.getCachedResult("crude-oil", "accuracy_weighted")
      ).toBeUndefined();
      expect(
        result.current.getCachedResult("bitcoin", "top_k_sharpe")
      ).toBeUndefined();
      expect(
        result.current.getCachedResult("gold", "ridge_stacking")
      ).toBeUndefined();
    });

    it("works when cache is already empty", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      expect(result.current.methodResults).toEqual({});

      act(() => {
        result.current.clearCache();
      });

      expect(result.current.methodResults).toEqual({});
    });

    it("allows caching again after clear", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const result1 = createMockResult("accuracy_weighted", "natural-gas");

      act(() => {
        result.current.cacheResult("natural-gas", "accuracy_weighted", result1);
      });

      act(() => {
        result.current.clearCache();
      });

      const result2 = createMockResult("exponential_decay", "natural-gas");

      act(() => {
        result.current.cacheResult("natural-gas", "exponential_decay", result2);
      });

      expect(result.current.methodResults).toEqual({
        "natural-gas": {
          exponential_decay: result2,
        },
      });
    });

    it("does not affect currentMethod", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      act(() => {
        result.current.setMethod("ridge_stacking");
      });

      const mockResult = createMockResult("ridge_stacking", "copper");

      act(() => {
        result.current.cacheResult("copper", "ridge_stacking", mockResult);
      });

      act(() => {
        result.current.clearCache();
      });

      expect(result.current.currentMethod).toBe("ridge_stacking");
    });

    it("maintains callback stability across renders", () => {
      const { result, rerender } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const clearCacheRef1 = result.current.clearCache;
      rerender();
      const clearCacheRef2 = result.current.clearCache;

      expect(clearCacheRef1).toBe(clearCacheRef2);
    });
  });

  describe("useEnsembleContext hook", () => {
    it("throws error when used outside provider", () => {
      // Suppress console.error for this test since we expect an error
      const consoleSpy = jest
        .spyOn(console, "error")
        .mockImplementation(() => {});

      expect(() => {
        renderHook(() => useEnsembleContext());
      }).toThrow("useEnsembleContext must be used within an EnsembleProvider");

      consoleSpy.mockRestore();
    });

    it("returns context value when inside provider", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      expect(result.current).toHaveProperty("currentMethod");
      expect(result.current).toHaveProperty("setMethod");
      expect(result.current).toHaveProperty("methodResults");
      expect(result.current).toHaveProperty("cacheResult");
      expect(result.current).toHaveProperty("getCachedResult");
      expect(result.current).toHaveProperty("clearCache");
    });
  });

  describe("useEnsembleMethod hook", () => {
    it("throws error when used outside provider", () => {
      const consoleSpy = jest
        .spyOn(console, "error")
        .mockImplementation(() => {});

      expect(() => {
        renderHook(() => useEnsembleMethod());
      }).toThrow("useEnsembleContext must be used within an EnsembleProvider");

      consoleSpy.mockRestore();
    });

    it("returns tuple of [currentMethod, setMethod]", () => {
      const { result } = renderHook(() => useEnsembleMethod(), {
        wrapper: createWrapper(),
      });

      expect(Array.isArray(result.current)).toBe(true);
      expect(result.current.length).toBe(2);
      expect(result.current[0]).toBe("top_k_sharpe");
      expect(typeof result.current[1]).toBe("function");
    });

    it("setMethod from tuple updates method correctly", () => {
      const { result } = renderHook(() => useEnsembleMethod(), {
        wrapper: createWrapper(),
      });

      act(() => {
        result.current[1]("pairwise_slope");
      });

      expect(result.current[0]).toBe("pairwise_slope");
    });

    it("respects custom default method", () => {
      const { result } = renderHook(() => useEnsembleMethod(), {
        wrapper: createWrapper("exponential_decay"),
      });

      expect(result.current[0]).toBe("exponential_decay");
    });
  });

  describe("context value memoization", () => {
    it("memoizes context value when dependencies unchanged", () => {
      const { result, rerender } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const value1 = result.current;
      rerender();
      const value2 = result.current;

      // Context value should be the same reference if nothing changed
      // Note: This tests useMemo behavior
      expect(value1.currentMethod).toBe(value2.currentMethod);
    });

    it("updates context value when state changes", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const initialMethod = result.current.currentMethod;

      act(() => {
        result.current.setMethod("inverse_variance");
      });

      expect(result.current.currentMethod).not.toBe(initialMethod);
      expect(result.current.currentMethod).toBe("inverse_variance");
    });
  });

  describe("edge cases", () => {
    it("handles rapid method changes", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      act(() => {
        result.current.setMethod("accuracy_weighted");
        result.current.setMethod("exponential_decay");
        result.current.setMethod("top_k_sharpe");
        result.current.setMethod("ridge_stacking");
        result.current.setMethod("inverse_variance");
        result.current.setMethod("pairwise_slope");
      });

      expect(result.current.currentMethod).toBe("pairwise_slope");
    });

    it("handles caching with special characters in symbol names", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const mockResult = createMockResult("accuracy_weighted", "test-asset");

      act(() => {
        result.current.cacheResult(
          "test-asset-with-dashes",
          "accuracy_weighted",
          mockResult
        );
      });

      expect(
        result.current.getCachedResult(
          "test-asset-with-dashes",
          "accuracy_weighted"
        )
      ).toEqual(mockResult);
    });

    it("handles empty string symbol gracefully", () => {
      const { result } = renderHook(() => useEnsembleContext(), {
        wrapper: createWrapper(),
      });

      const mockResult = createMockResult("accuracy_weighted", "");

      act(() => {
        result.current.cacheResult("", "accuracy_weighted", mockResult);
      });

      expect(result.current.getCachedResult("", "accuracy_weighted")).toEqual(
        mockResult
      );
    });
  });
});
