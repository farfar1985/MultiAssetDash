import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SignalCard } from "../SignalCard";
import type { AssetData, SignalData } from "@/lib/api-client";
import type { AssetId, SignalDirection } from "@/types";

// Mock the useApi hooks
jest.mock("@/hooks/useApi", () => ({
  useAsset: jest.fn(),
  useSignal: jest.fn(),
}));

// Mock Radix UI Portal for Select dropdown
jest.mock("@radix-ui/react-select", () => {
  const actual = jest.requireActual("@radix-ui/react-select");
  return {
    ...actual,
    Portal: ({ children }: { children: React.ReactNode }) => children,
  };
});

import { useAsset, useSignal } from "@/hooks/useApi";

const mockUseAsset = useAsset as jest.MockedFunction<typeof useAsset>;
const mockUseSignal = useSignal as jest.MockedFunction<typeof useSignal>;

// Mock asset data factory
function createMockAsset(overrides: Partial<AssetData> = {}): AssetData {
  return {
    id: "crude-oil" as AssetId,
    name: "Crude Oil",
    symbol: "CL",
    category: "energy",
    currentPrice: 75.42,
    change24h: 1.25,
    changePercent24h: 1.68,
    ...overrides,
  };
}

// Mock signal data factory
function createMockSignal(overrides: Partial<SignalData> = {}): SignalData {
  return {
    assetId: "crude-oil" as AssetId,
    direction: "bullish" as SignalDirection,
    confidence: 78,
    horizon: "D+1",
    modelsAgreeing: 7500,
    modelsTotal: 10000,
    sharpeRatio: 2.34,
    directionalAccuracy: 61.24,
    totalReturn: 53.4,
    generatedAt: new Date().toISOString(),
    ...overrides,
  };
}

// Helper to set up mock hook returns
function setupMocks(options: {
  asset?: AssetData | null;
  signal?: SignalData | null;
  isAssetLoading?: boolean;
  isSignalLoading?: boolean;
  assetError?: Error | null;
  signalError?: Error | null;
}) {
  mockUseAsset.mockReturnValue({
    data: options.asset ?? undefined,
    isLoading: options.isAssetLoading ?? false,
    error: options.assetError ?? null,
    isError: !!options.assetError,
    isPending: options.isAssetLoading ?? false,
    isSuccess: !options.isAssetLoading && !options.assetError && !!options.asset,
    isFetching: false,
    isRefetching: false,
    isStale: false,
    failureCount: 0,
    failureReason: null,
    refetch: jest.fn(),
    status: options.isAssetLoading ? "pending" : options.assetError ? "error" : "success",
    fetchStatus: "idle",
    dataUpdatedAt: Date.now(),
    errorUpdateCount: 0,
    errorUpdatedAt: 0,
    isLoadingError: false,
    isFetched: true,
    isFetchedAfterMount: true,
    isInitialLoading: false,
    isPaused: false,
    isPlaceholderData: false,
    isRefetchError: false,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any);

  mockUseSignal.mockReturnValue({
    data: options.signal ?? undefined,
    isLoading: options.isSignalLoading ?? false,
    error: options.signalError ?? null,
    isError: !!options.signalError,
    isPending: options.isSignalLoading ?? false,
    isSuccess: !options.isSignalLoading && !options.signalError && !!options.signal,
    isFetching: false,
    isRefetching: false,
    isStale: false,
    failureCount: 0,
    failureReason: null,
    refetch: jest.fn(),
    status: options.isSignalLoading ? "pending" : options.signalError ? "error" : "success",
    fetchStatus: "idle",
    dataUpdatedAt: Date.now(),
    errorUpdateCount: 0,
    errorUpdatedAt: 0,
    isLoadingError: false,
    isFetched: true,
    isFetchedAfterMount: true,
    isInitialLoading: false,
    isPaused: false,
    isPlaceholderData: false,
    isRefetchError: false,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any);
}

describe("SignalCard", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("loading state", () => {
    it("renders skeleton while asset is loading", () => {
      setupMocks({
        isAssetLoading: true,
        signal: createMockSignal(),
      });

      render(<SignalCard assetId="crude-oil" />);

      // Skeleton elements should be present (multiple skeleton items)
      const skeletons = document.querySelectorAll('[class*="bg-neutral-800"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });

    it("renders skeleton while signal is loading", () => {
      setupMocks({
        asset: createMockAsset(),
        isSignalLoading: true,
      });

      render(<SignalCard assetId="crude-oil" />);

      // Should show loading state
      const card = document.querySelector('[class*="bg-neutral-900"]');
      expect(card).toBeInTheDocument();
    });

    it("renders skeleton when both are loading", () => {
      setupMocks({
        isAssetLoading: true,
        isSignalLoading: true,
      });

      render(<SignalCard assetId="crude-oil" />);

      // Card should be in loading state
      expect(
        screen.queryByText(/Crude Oil/i)
      ).not.toBeInTheDocument();
    });
  });

  describe("error state", () => {
    it("renders error state when asset fetch fails", () => {
      setupMocks({
        assetError: new Error("Failed to fetch asset"),
        signal: createMockSignal(),
      });

      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("Failed to load signal")).toBeInTheDocument();
      expect(screen.getByText("Failed to fetch asset")).toBeInTheDocument();
    });

    it("renders error state when signal fetch fails", () => {
      setupMocks({
        asset: createMockAsset(),
        signalError: new Error("Network error"),
      });

      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("Failed to load signal")).toBeInTheDocument();
      expect(screen.getByText("Network error")).toBeInTheDocument();
    });

    it("shows asset error over signal error when both fail", () => {
      setupMocks({
        assetError: new Error("Asset error"),
        signalError: new Error("Signal error"),
      });

      render(<SignalCard assetId="crude-oil" />);

      // Asset error should be shown (it's checked first)
      expect(screen.getByText("Asset error")).toBeInTheDocument();
    });

    it("renders error card with correct styling", () => {
      setupMocks({
        assetError: new Error("Some error"),
      });

      render(<SignalCard assetId="crude-oil" />);

      const errorCard = document.querySelector('[class*="border-red-900"]');
      expect(errorCard).toBeInTheDocument();
    });
  });

  describe("null data state", () => {
    it("returns null when asset is null after loading", () => {
      setupMocks({
        asset: null,
        signal: createMockSignal(),
      });

      const { container } = render(<SignalCard assetId="crude-oil" />);

      expect(container.firstChild).toBeNull();
    });

    it("returns null when signal is null after loading", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: null,
      });

      const { container } = render(<SignalCard assetId="crude-oil" />);

      expect(container.firstChild).toBeNull();
    });
  });

  describe("successful data rendering", () => {
    beforeEach(() => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal(),
      });
    });

    it("displays asset symbol", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("CL")).toBeInTheDocument();
    });

    it("displays asset name", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("Crude Oil")).toBeInTheDocument();
    });

    it("displays current price formatted correctly", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("$75.42")).toBeInTheDocument();
    });

    it("displays positive 24h change with plus sign and green color", () => {
      render(<SignalCard assetId="crude-oil" />);

      const changeElement = screen.getByText("+1.68%");
      expect(changeElement).toBeInTheDocument();
      expect(changeElement).toHaveClass("text-green-500");
    });

    it("displays negative 24h change with red color", () => {
      setupMocks({
        asset: createMockAsset({ changePercent24h: -2.5 }),
        signal: createMockSignal(),
      });

      render(<SignalCard assetId="crude-oil" />);

      const changeElement = screen.getByText("-2.50%");
      expect(changeElement).toBeInTheDocument();
      expect(changeElement).toHaveClass("text-red-500");
    });

    it("displays signal direction badge", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("bullish")).toBeInTheDocument();
    });

    it("displays confidence percentage", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("78%")).toBeInTheDocument();
    });

    it("displays model agreement text", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(
        screen.getByText("7,500 / 10,000 models agree")
      ).toBeInTheDocument();
    });

    it("displays sharpe ratio with 2 decimal places", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("2.34")).toBeInTheDocument();
    });

    it("displays directional accuracy with 1 decimal place", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("61.2%")).toBeInTheDocument();
    });

    it("displays total return with 1 decimal place and plus sign for positive", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("+53.4%")).toBeInTheDocument();
    });
  });

  describe("direction colors and icons", () => {
    it("displays green styling for bullish direction", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ direction: "bullish" }),
      });

      render(<SignalCard assetId="crude-oil" />);

      // The Badge component renders as a div
      const badge = screen.getByText("bullish").closest("div");
      expect(badge).toHaveClass("bg-green-500/10");
      expect(badge).toHaveClass("text-green-500");
    });

    it("displays up arrow icon for bullish direction", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ direction: "bullish" }),
      });

      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("↑")).toBeInTheDocument();
    });

    it("displays red styling for bearish direction", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ direction: "bearish" }),
      });

      render(<SignalCard assetId="crude-oil" />);

      // The Badge component renders as a div
      const badge = screen.getByText("bearish").closest("div");
      expect(badge).toHaveClass("bg-red-500/10");
      expect(badge).toHaveClass("text-red-500");
    });

    it("displays down arrow icon for bearish direction", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ direction: "bearish" }),
      });

      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("↓")).toBeInTheDocument();
    });

    it("displays yellow styling for neutral direction", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ direction: "neutral" }),
      });

      render(<SignalCard assetId="crude-oil" />);

      // The Badge component renders as a div
      const badge = screen.getByText("neutral").closest("div");
      expect(badge).toHaveClass("bg-yellow-500/10");
      expect(badge).toHaveClass("text-yellow-500");
    });

    it("displays right arrow icon for neutral direction", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ direction: "neutral" }),
      });

      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("→")).toBeInTheDocument();
    });
  });

  describe("confidence bar colors", () => {
    it("uses green color for confidence >= 80", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ confidence: 85 }),
      });

      render(<SignalCard assetId="crude-oil" />);

      const progressBar = document.querySelector('[class*="rounded-full"][style*="width"]');
      expect(progressBar).toHaveClass("bg-green-500");
    });

    it("uses blue color for confidence >= 60 and < 80", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ confidence: 70 }),
      });

      render(<SignalCard assetId="crude-oil" />);

      const progressBar = document.querySelector('[class*="rounded-full"][style*="width"]');
      expect(progressBar).toHaveClass("bg-blue-500");
    });

    it("uses yellow color for confidence >= 40 and < 60", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ confidence: 50 }),
      });

      render(<SignalCard assetId="crude-oil" />);

      const progressBar = document.querySelector('[class*="rounded-full"][style*="width"]');
      expect(progressBar).toHaveClass("bg-yellow-500");
    });

    it("uses red color for confidence < 40", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ confidence: 30 }),
      });

      render(<SignalCard assetId="crude-oil" />);

      const progressBar = document.querySelector('[class*="rounded-full"][style*="width"]');
      expect(progressBar).toHaveClass("bg-red-500");
    });

    it("sets correct width based on confidence", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ confidence: 65 }),
      });

      render(<SignalCard assetId="crude-oil" />);

      const progressBar = document.querySelector('[class*="rounded-full"][style*="width"]');
      expect(progressBar).toHaveStyle({ width: "65%" });
    });
  });

  describe("return value styling", () => {
    it("displays positive return in green", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ totalReturn: 25.5 }),
      });

      render(<SignalCard assetId="crude-oil" />);

      const returnElement = screen.getByText("+25.5%");
      expect(returnElement).toHaveClass("text-green-500");
    });

    it("displays negative return in red", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ totalReturn: -15.2 }),
      });

      render(<SignalCard assetId="crude-oil" />);

      const returnElement = screen.getByText("-15.2%");
      expect(returnElement).toHaveClass("text-red-500");
    });

    it("displays zero return in green with plus sign", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ totalReturn: 0 }),
      });

      render(<SignalCard assetId="crude-oil" />);

      const returnElement = screen.getByText("+0.0%");
      expect(returnElement).toHaveClass("text-green-500");
    });
  });

  describe("price formatting", () => {
    it("formats BTC price with dollar sign and 2 decimals", () => {
      setupMocks({
        asset: createMockAsset({
          id: "bitcoin" as AssetId,
          name: "Bitcoin",
          symbol: "BTC",
          currentPrice: 45678.9,
        }),
        signal: createMockSignal({ assetId: "bitcoin" as AssetId }),
      });

      render(<SignalCard assetId="bitcoin" />);

      expect(screen.getByText("$45,678.90")).toBeInTheDocument();
    });

    it("formats price >= 100 with dollar sign and 2 decimals", () => {
      setupMocks({
        asset: createMockAsset({ currentPrice: 1234.567 }),
        signal: createMockSignal(),
      });

      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("$1,234.57")).toBeInTheDocument();
    });

    it("formats price < 100 with dollar sign and 2 decimals", () => {
      setupMocks({
        asset: createMockAsset({ currentPrice: 45.6789 }),
        signal: createMockSignal(),
      });

      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("$45.68")).toBeInTheDocument();
    });
  });

  describe("horizon selector", () => {
    it("displays default horizon D+1", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ horizon: "D+1" }),
      });

      render(<SignalCard assetId="crude-oil" />);

      const trigger = screen.getByRole("combobox");
      expect(trigger).toHaveTextContent("D+1");
    });

    it("uses custom default horizon when provided", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ horizon: "D+5" }),
      });

      render(<SignalCard assetId="crude-oil" defaultHorizon="D+5" />);

      // The useSignal hook should be called with D+5
      expect(mockUseSignal).toHaveBeenCalledWith("crude-oil", "D+5");
    });

    it("shows all horizon options when clicked", async () => {
      const user = userEvent.setup();
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal(),
      });

      render(<SignalCard assetId="crude-oil" />);

      const trigger = screen.getByRole("combobox");
      await user.click(trigger);

      expect(screen.getByRole("option", { name: "D+1" })).toBeInTheDocument();
      expect(screen.getByRole("option", { name: "D+5" })).toBeInTheDocument();
      expect(screen.getByRole("option", { name: "D+10" })).toBeInTheDocument();
    });

    it("calls useSignal with new horizon when changed", async () => {
      const user = userEvent.setup();
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal(),
      });

      render(<SignalCard assetId="crude-oil" />);

      const trigger = screen.getByRole("combobox");
      await user.click(trigger);

      const d5Option = screen.getByRole("option", { name: "D+5" });
      await user.click(d5Option);

      // Hook should be called with new horizon
      expect(mockUseSignal).toHaveBeenCalledWith("crude-oil", "D+5");
    });
  });

  describe("onClick handler", () => {
    it("does not have cursor-pointer class when onClick not provided", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal(),
      });

      render(<SignalCard assetId="crude-oil" />);

      const card = document.querySelector('[class*="bg-neutral-900"]');
      expect(card).not.toHaveClass("cursor-pointer");
    });

    it("has cursor-pointer class when onClick is provided", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal(),
      });

      render(<SignalCard assetId="crude-oil" onClick={() => {}} />);

      const card = document.querySelector('[class*="cursor-pointer"]');
      expect(card).toBeInTheDocument();
    });

    it("calls onClick when card is clicked", async () => {
      const user = userEvent.setup();
      const handleClick = jest.fn();

      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal(),
      });

      render(<SignalCard assetId="crude-oil" onClick={handleClick} />);

      const card = document.querySelector('[class*="bg-neutral-900"]')!;
      await user.click(card);

      expect(handleClick).toHaveBeenCalledTimes(1);
    });
  });

  describe("edge cases", () => {
    it("handles very long asset names", () => {
      setupMocks({
        asset: createMockAsset({
          name: "Very Long Asset Name That Might Overflow The Container",
        }),
        signal: createMockSignal(),
      });

      render(<SignalCard assetId="crude-oil" />);

      expect(
        screen.getByText("Very Long Asset Name That Might Overflow The Container")
      ).toBeInTheDocument();
    });

    it("handles zero price change", () => {
      setupMocks({
        asset: createMockAsset({ changePercent24h: 0 }),
        signal: createMockSignal(),
      });

      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("+0.00%")).toBeInTheDocument();
    });

    it("handles large model counts with proper formatting", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({
          modelsAgreeing: 9500,
          modelsTotal: 10179,
        }),
      });

      render(<SignalCard assetId="crude-oil" />);

      expect(
        screen.getByText("9,500 / 10,179 models agree")
      ).toBeInTheDocument();
    });

    it("handles boundary confidence value of 80", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ confidence: 80 }),
      });

      render(<SignalCard assetId="crude-oil" />);

      const progressBar = document.querySelector('[class*="rounded-full"][style*="width"]');
      expect(progressBar).toHaveClass("bg-green-500");
    });

    it("handles boundary confidence value of 60", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ confidence: 60 }),
      });

      render(<SignalCard assetId="crude-oil" />);

      const progressBar = document.querySelector('[class*="rounded-full"][style*="width"]');
      expect(progressBar).toHaveClass("bg-blue-500");
    });

    it("handles boundary confidence value of 40", () => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal({ confidence: 40 }),
      });

      render(<SignalCard assetId="crude-oil" />);

      const progressBar = document.querySelector('[class*="rounded-full"][style*="width"]');
      expect(progressBar).toHaveClass("bg-yellow-500");
    });
  });

  describe("different asset types", () => {
    const assets: Array<{ id: AssetId; symbol: string; name: string }> = [
      { id: "crude-oil", symbol: "CL", name: "Crude Oil" },
      { id: "bitcoin", symbol: "BTC", name: "Bitcoin" },
      { id: "gold", symbol: "GC", name: "Gold" },
      { id: "silver", symbol: "SI", name: "Silver" },
      { id: "natural-gas", symbol: "NG", name: "Natural Gas" },
      { id: "copper", symbol: "HG", name: "Copper" },
      { id: "wheat", symbol: "ZW", name: "Wheat" },
      { id: "corn", symbol: "ZC", name: "Corn" },
      { id: "soybean", symbol: "ZS", name: "Soybean" },
      { id: "platinum", symbol: "PL", name: "Platinum" },
    ];

    assets.forEach(({ id, symbol, name }) => {
      it(`renders correctly for ${name}`, () => {
        setupMocks({
          asset: createMockAsset({ id, symbol, name }),
          signal: createMockSignal({ assetId: id }),
        });

        render(<SignalCard assetId={id} />);

        expect(screen.getByText(symbol)).toBeInTheDocument();
        expect(screen.getByText(name)).toBeInTheDocument();
      });
    });
  });

  describe("metric labels", () => {
    beforeEach(() => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal(),
      });
    });

    it("displays Sharpe label", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("Sharpe")).toBeInTheDocument();
    });

    it("displays DA% label", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("DA%")).toBeInTheDocument();
    });

    it("displays Return label", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("Return")).toBeInTheDocument();
    });

    it("displays Confidence label", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByText("Confidence")).toBeInTheDocument();
    });
  });

  describe("accessibility", () => {
    beforeEach(() => {
      setupMocks({
        asset: createMockAsset(),
        signal: createMockSignal(),
      });
    });

    it("horizon selector has combobox role", () => {
      render(<SignalCard assetId="crude-oil" />);

      expect(screen.getByRole("combobox")).toBeInTheDocument();
    });

    it("horizon options have option role", async () => {
      const user = userEvent.setup();
      render(<SignalCard assetId="crude-oil" />);

      const trigger = screen.getByRole("combobox");
      await user.click(trigger);

      expect(screen.getAllByRole("option")).toHaveLength(3);
    });
  });
});
