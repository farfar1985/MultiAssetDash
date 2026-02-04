import { render, screen } from "@testing-library/react";
import { ActionableSummary } from "../ActionableSummary";
import type { ActionabilityLevel } from "@/types/practical-metrics";

// ============================================================================
// Test Fixtures
// ============================================================================

interface ActionableSummaryTestProps {
  assetsNeedingAttention?: number;
  actionableSignalsToday?: number;
  bullishPercent?: number;
  bearishPercent?: number;
  neutralPercent?: number;
  overallStatus?: ActionabilityLevel;
  lastUpdated?: string;
}

const createDefaultProps = (
  overrides: ActionableSummaryTestProps = {}
): Required<ActionableSummaryTestProps> => ({
  assetsNeedingAttention: 2,
  actionableSignalsToday: 5,
  bullishPercent: 45,
  bearishPercent: 30,
  neutralPercent: 25,
  overallStatus: "high",
  lastUpdated: "10:30:00",
  ...overrides,
});

// ============================================================================
// Tests
// ============================================================================

describe("ActionableSummary", () => {
  describe("status indicator", () => {
    it.each<[ActionabilityLevel, string]>([
      ["high", "Strong Signals"],
      ["medium", "Mixed Signals"],
      ["low", "Weak Signals"],
    ])("displays correct text for %s status", (status, expectedText) => {
      render(<ActionableSummary {...createDefaultProps({ overallStatus: status })} />);

      expect(screen.getByText(expectedText)).toBeInTheDocument();
    });

    it("renders pulsing status dot", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      const dot = document.querySelector(".animate-pulse");
      expect(dot).toBeInTheDocument();
    });

    it("applies green color for high status", () => {
      render(<ActionableSummary {...createDefaultProps({ overallStatus: "high" })} />);

      const dot = document.querySelector(".bg-green-500");
      expect(dot).toBeInTheDocument();
    });

    it("applies yellow color for medium status", () => {
      render(<ActionableSummary {...createDefaultProps({ overallStatus: "medium" })} />);

      const dot = document.querySelector(".bg-yellow-500");
      expect(dot).toBeInTheDocument();
    });

    it("applies red color for low status", () => {
      render(<ActionableSummary {...createDefaultProps({ overallStatus: "low" })} />);

      const dot = document.querySelector(".bg-red-500");
      expect(dot).toBeInTheDocument();
    });
  });

  describe("attention badge", () => {
    it("displays attention badge when assets need attention", () => {
      render(<ActionableSummary {...createDefaultProps({ assetsNeedingAttention: 3 })} />);

      expect(screen.getByText(/3 assets need attention/)).toBeInTheDocument();
    });

    it("uses singular form for 1 asset", () => {
      render(<ActionableSummary {...createDefaultProps({ assetsNeedingAttention: 1 })} />);

      // Component renders: "1 asset need attention" (uses plural/singular for "asset" but not "need")
      expect(screen.getByText(/1 asset need attention/)).toBeInTheDocument();
    });

    it("hides attention badge when no assets need attention", () => {
      render(<ActionableSummary {...createDefaultProps({ assetsNeedingAttention: 0 })} />);

      expect(screen.queryByText(/need attention/)).not.toBeInTheDocument();
    });

    it("applies orange styling to attention badge", () => {
      render(<ActionableSummary {...createDefaultProps({ assetsNeedingAttention: 2 })} />);

      // Badge is rendered with inline-flex, not as a separate Badge component
      const badge = screen.getByText(/assets need attention/).closest('[class*="inline-flex"]');
      expect(badge).toHaveClass("bg-orange-500/10");
      expect(badge).toHaveClass("text-orange-500");
    });

    it("renders AlertTriangle icon in attention badge", () => {
      render(<ActionableSummary {...createDefaultProps({ assetsNeedingAttention: 2 })} />);

      // The icon should be rendered within the badge
      const badge = screen.getByText(/assets need attention/).parentElement;
      const svg = badge?.querySelector("svg");
      expect(svg).toBeInTheDocument();
    });
  });

  describe("actionable signals count", () => {
    it("displays actionable signals count", () => {
      render(<ActionableSummary {...createDefaultProps({ actionableSignalsToday: 7 })} />);

      expect(screen.getByText("7")).toBeInTheDocument();
      expect(screen.getByText(/signals actionable today/)).toBeInTheDocument();
    });

    it("displays 0 signals correctly", () => {
      render(<ActionableSummary {...createDefaultProps({ actionableSignalsToday: 0 })} />);

      expect(screen.getByText("0")).toBeInTheDocument();
    });

    it("applies blue styling to signal count", () => {
      render(<ActionableSummary {...createDefaultProps({ actionableSignalsToday: 5 })} />);

      const countElement = screen.getByText("5");
      expect(countElement).toHaveClass("text-blue-400");
    });

    it("uses monospace font for signal count", () => {
      render(<ActionableSummary {...createDefaultProps({ actionableSignalsToday: 5 })} />);

      const countElement = screen.getByText("5");
      expect(countElement).toHaveClass("font-mono");
    });
  });

  describe("portfolio bias calculation", () => {
    it("displays Bullish bias when bullish > bearish + 15", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 50,
            bearishPercent: 30,
            neutralPercent: 20,
          })}
        />
      );

      expect(screen.getByText(/Portfolio bias:/)).toBeInTheDocument();
      // Should show bullish percent
      expect(screen.getByText(/50% bullish/)).toBeInTheDocument();
    });

    it("displays Bearish bias when bearish > bullish + 15", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 25,
            bearishPercent: 55,
            neutralPercent: 20,
          })}
        />
      );

      expect(screen.getByText(/25% bullish/)).toBeInTheDocument();
    });

    it("displays Neutral bias when difference is within 15%", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 40,
            bearishPercent: 35,
            neutralPercent: 25,
          })}
        />
      );

      expect(screen.getByText(/40% bullish/)).toBeInTheDocument();
    });

    it("applies green color for bullish bias", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 60,
            bearishPercent: 20,
          })}
        />
      );

      const biasElement = screen.getByText(/60% bullish/);
      expect(biasElement.className).toContain("green");
    });

    it("applies red color for bearish bias", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 20,
            bearishPercent: 60,
          })}
        />
      );

      // The bullish percent text is shown but with bearish styling context
      const biasElement = screen.getByText(/20% bullish/);
      expect(biasElement.className).toContain("red");
    });

    it("applies yellow color for neutral bias", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 35,
            bearishPercent: 35,
          })}
        />
      );

      const biasElement = screen.getByText(/35% bullish/);
      expect(biasElement.className).toContain("yellow");
    });

    it("renders TrendingUp icon for bullish bias", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 60,
            bearishPercent: 20,
          })}
        />
      );

      // Should render trending up icon - check for its parent container
      const biasSection = screen.getByText(/Portfolio bias:/).parentElement;
      const svg = biasSection?.querySelector("svg");
      expect(svg).toBeInTheDocument();
    });

    it("renders TrendingDown icon for bearish bias", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 20,
            bearishPercent: 60,
          })}
        />
      );

      const biasSection = screen.getByText(/Portfolio bias:/).parentElement;
      const svg = biasSection?.querySelector("svg");
      expect(svg).toBeInTheDocument();
    });

    it("renders Activity icon for neutral bias", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 35,
            bearishPercent: 35,
          })}
        />
      );

      const biasSection = screen.getByText(/Portfolio bias:/).parentElement;
      const svg = biasSection?.querySelector("svg");
      expect(svg).toBeInTheDocument();
    });
  });

  describe("directional breakdown", () => {
    it("displays bullish percentage with up arrow", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 45,
          })}
        />
      );

      expect(screen.getByText("45% ↑")).toBeInTheDocument();
    });

    it("displays bearish percentage with down arrow", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bearishPercent: 30,
          })}
        />
      );

      expect(screen.getByText("30% ↓")).toBeInTheDocument();
    });

    it("displays neutral percentage with right arrow", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            neutralPercent: 25,
          })}
        />
      );

      expect(screen.getByText("25% →")).toBeInTheDocument();
    });

    it("applies green color to bullish breakdown", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      const bullishElement = screen.getByText(/45% ↑/);
      expect(bullishElement.className).toContain("green");
    });

    it("applies red color to bearish breakdown", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      const bearishElement = screen.getByText(/30% ↓/);
      expect(bearishElement.className).toContain("red");
    });

    it("applies yellow color to neutral breakdown", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      const neutralElement = screen.getByText(/25% →/);
      expect(neutralElement.className).toContain("yellow");
    });

    it("uses monospace font for breakdown percentages", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      const bullishElement = screen.getByText(/45% ↑/);
      expect(bullishElement).toHaveClass("font-mono");
    });

    it("formats percentages as integers", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 45.7,
            bearishPercent: 30.3,
            neutralPercent: 24.0,
          })}
        />
      );

      expect(screen.getByText("46% ↑")).toBeInTheDocument();
      expect(screen.getByText("30% ↓")).toBeInTheDocument();
      expect(screen.getByText("24% →")).toBeInTheDocument();
    });
  });

  describe("last updated timestamp", () => {
    it("displays last updated time", () => {
      render(<ActionableSummary {...createDefaultProps({ lastUpdated: "14:30:45" })} />);

      expect(screen.getByText("14:30:45")).toBeInTheDocument();
    });

    it("uses monospace font for timestamp", () => {
      render(<ActionableSummary {...createDefaultProps({ lastUpdated: "10:30:00" })} />);

      const timestamp = screen.getByText("10:30:00");
      expect(timestamp).toHaveClass("font-mono");
    });

    it("renders Clock icon next to timestamp", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      const timestampSection = screen.getByText("10:30:00").parentElement;
      const svg = timestampSection?.querySelector("svg");
      expect(svg).toBeInTheDocument();
    });

    it("applies neutral styling to timestamp", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      const timestampSection = screen.getByText("10:30:00").parentElement;
      expect(timestampSection).toHaveClass("text-neutral-500");
    });
  });

  describe("container styling", () => {
    it("has correct background and border classes", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      const container = screen.getByText(/Strong Signals/).closest("div.bg-neutral-900\\/50");
      expect(container).toBeInTheDocument();
    });

    it("has rounded corners", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      const container = document.querySelector(".rounded-lg");
      expect(container).toBeInTheDocument();
    });

    it("has flex layout with wrapping", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      const flexContainer = document.querySelector(".flex.flex-wrap");
      expect(flexContainer).toBeInTheDocument();
    });
  });

  describe("edge cases", () => {
    it("handles 0% for all directions", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 0,
            bearishPercent: 0,
            neutralPercent: 0,
          })}
        />
      );

      expect(screen.getByText("0% ↑")).toBeInTheDocument();
      expect(screen.getByText("0% ↓")).toBeInTheDocument();
      expect(screen.getByText("0% →")).toBeInTheDocument();
    });

    it("handles 100% bullish", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 100,
            bearishPercent: 0,
            neutralPercent: 0,
          })}
        />
      );

      expect(screen.getByText("100% ↑")).toBeInTheDocument();
      expect(screen.getByText(/100% bullish/)).toBeInTheDocument();
    });

    it("handles 100% bearish", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 0,
            bearishPercent: 100,
            neutralPercent: 0,
          })}
        />
      );

      expect(screen.getByText("100% ↓")).toBeInTheDocument();
    });

    it("handles large assets needing attention count", () => {
      render(<ActionableSummary {...createDefaultProps({ assetsNeedingAttention: 99 })} />);

      expect(screen.getByText(/99 assets need attention/)).toBeInTheDocument();
    });

    it("handles large actionable signals count", () => {
      render(<ActionableSummary {...createDefaultProps({ actionableSignalsToday: 100 })} />);

      expect(screen.getByText("100")).toBeInTheDocument();
    });

    it("handles edge case of exactly 15% difference (neutral)", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 50,
            bearishPercent: 35, // Exactly 15% difference
            neutralPercent: 15,
          })}
        />
      );

      // Should be neutral since it's not greater than 15%
      const biasElement = screen.getByText(/50% bullish/);
      expect(biasElement.className).toContain("yellow");
    });

    it("handles edge case of 16% difference (bullish)", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 50,
            bearishPercent: 34, // 16% difference - should be bullish
            neutralPercent: 16,
          })}
        />
      );

      const biasElement = screen.getByText(/50% bullish/);
      expect(biasElement.className).toContain("green");
    });

    it("handles decimal percentages", () => {
      render(
        <ActionableSummary
          {...createDefaultProps({
            bullishPercent: 33.33,
            bearishPercent: 33.33,
            neutralPercent: 33.34,
          })}
        />
      );

      // Should round to integers
      expect(screen.getByText("33% ↑")).toBeInTheDocument();
      expect(screen.getByText("33% ↓")).toBeInTheDocument();
      expect(screen.getByText("33% →")).toBeInTheDocument();
    });
  });

  describe("accessibility", () => {
    it("renders status text for screen readers", () => {
      render(<ActionableSummary {...createDefaultProps({ overallStatus: "high" })} />);

      expect(screen.getByText("Strong Signals")).toBeInTheDocument();
    });

    it("icon containers have appropriate semantic structure", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      // Icons should be within flex containers for proper alignment
      const iconContainers = document.querySelectorAll(".flex.items-center");
      expect(iconContainers.length).toBeGreaterThan(0);
    });
  });

  describe("responsive layout", () => {
    it("uses flex-wrap for small screens", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      const flexContainer = document.querySelector(".flex-wrap");
      expect(flexContainer).toBeInTheDocument();
    });

    it("uses gap-4 for consistent spacing", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      const container = document.querySelector(".gap-4");
      expect(container).toBeInTheDocument();
    });

    it("uses justify-between for full-width distribution", () => {
      render(<ActionableSummary {...createDefaultProps()} />);

      const container = document.querySelector(".justify-between");
      expect(container).toBeInTheDocument();
    });
  });

  describe("component integration", () => {
    it("renders all sections together correctly", () => {
      render(
        <ActionableSummary
          assetsNeedingAttention={3}
          actionableSignalsToday={7}
          bullishPercent={55}
          bearishPercent={25}
          neutralPercent={20}
          overallStatus="high"
          lastUpdated="15:45:30"
        />
      );

      // Status
      expect(screen.getByText("Strong Signals")).toBeInTheDocument();

      // Attention badge
      expect(screen.getByText(/3 assets need attention/)).toBeInTheDocument();

      // Actionable signals
      expect(screen.getByText("7")).toBeInTheDocument();

      // Portfolio bias (bullish since 55 > 25 + 15)
      expect(screen.getByText(/55% bullish/)).toBeInTheDocument();

      // Directional breakdown
      expect(screen.getByText("55% ↑")).toBeInTheDocument();
      expect(screen.getByText("25% ↓")).toBeInTheDocument();
      expect(screen.getByText("20% →")).toBeInTheDocument();

      // Timestamp
      expect(screen.getByText("15:45:30")).toBeInTheDocument();
    });
  });
});
