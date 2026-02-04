"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useMemo,
  type ReactNode,
} from "react";
import type { EnsembleMethod, EnsembleResult } from "@/lib/api-client";

interface MethodResults {
  [symbol: string]: {
    [method in EnsembleMethod]?: EnsembleResult;
  };
}

interface EnsembleContextValue {
  currentMethod: EnsembleMethod;
  setMethod: (method: EnsembleMethod) => void;
  methodResults: MethodResults;
  cacheResult: (symbol: string, method: EnsembleMethod, result: EnsembleResult) => void;
  getCachedResult: (symbol: string, method: EnsembleMethod) => EnsembleResult | undefined;
  clearCache: () => void;
}

const EnsembleContext = createContext<EnsembleContextValue | null>(null);

interface EnsembleProviderProps {
  children: ReactNode;
  defaultMethod?: EnsembleMethod;
}

export function EnsembleProvider({
  children,
  defaultMethod = "top_k_sharpe",
}: EnsembleProviderProps) {
  const [currentMethod, setCurrentMethod] = useState<EnsembleMethod>(defaultMethod);
  const [methodResults, setMethodResults] = useState<MethodResults>({});

  const setMethod = useCallback((method: EnsembleMethod) => {
    setCurrentMethod(method);
  }, []);

  const cacheResult = useCallback(
    (symbol: string, method: EnsembleMethod, result: EnsembleResult) => {
      setMethodResults((prev) => ({
        ...prev,
        [symbol]: {
          ...prev[symbol],
          [method]: result,
        },
      }));
    },
    []
  );

  const getCachedResult = useCallback(
    (symbol: string, method: EnsembleMethod): EnsembleResult | undefined => {
      return methodResults[symbol]?.[method];
    },
    [methodResults]
  );

  const clearCache = useCallback(() => {
    setMethodResults({});
  }, []);

  const value = useMemo<EnsembleContextValue>(
    () => ({
      currentMethod,
      setMethod,
      methodResults,
      cacheResult,
      getCachedResult,
      clearCache,
    }),
    [currentMethod, setMethod, methodResults, cacheResult, getCachedResult, clearCache]
  );

  return (
    <EnsembleContext.Provider value={value}>
      {children}
    </EnsembleContext.Provider>
  );
}

export function useEnsembleContext(): EnsembleContextValue {
  const context = useContext(EnsembleContext);
  if (!context) {
    throw new Error("useEnsembleContext must be used within an EnsembleProvider");
  }
  return context;
}

export function useEnsembleMethod(): [EnsembleMethod, (method: EnsembleMethod) => void] {
  const { currentMethod, setMethod } = useEnsembleContext();
  return [currentMethod, setMethod];
}
