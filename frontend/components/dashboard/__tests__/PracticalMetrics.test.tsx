import { render, screen } from "@testing-library/react";
import { PracticalMetrics } from "../PracticalMetrics";
import type { PracticalMetricsData, ActionabilityLevel } from "@/types/practical-metrics";
import type { AssetId } from "@/types";
import type { Horizon } from "@/types/horizon-pairs";

// ============================================================================
// Test Fixtures
// ============================================================================

const createMockPracticalMetricsData = (
  overrides: Partial<{
    asset: AssetId;
    actionability: ActionabilityLevel;
    score: number;
    isActionable: boolean;
    direction: "up" | "down";
    predictedMove: number;
    confidence: number;
    bigMoveWinRate: number;
    positionSize: number;
    urgency: "immediate" | "today" | "this_week" | "monitor";
    coveredHorizons: Horizon[];
    optimalHorizon: Horizon | null;
    traditionalSharpe: number;
    overallWinRate: number;
    reasons: string[];
  }> = {}
): PracticalMetricsData => {
  const {
    asset = "crude-oil",
    actionability = "high",
    score = 78,
    isActionable = true,
    direction = "up",
    predictedMove = 2.5,
    confidence = 75,
    bigMoveWinRate = 65,
    positionSize = 60,
    urgency = "immediate",
    coveredHorizons = ["D+1", "D+2", "D+3", "D+5"],
    optimalHorizon = "D+3",
    traditionalSharpe = 1.8,
    overallWinRate = 58,
    reasons = ["Large predicted move", "High confidence", "Good horizon coverage"],
  } = overrides;

  return {
    asset,
    currentPrice: 75.5,
    signal: {
      isActionable,
      reasons,
      magnitude: {
        predictedMove: direction === "up" ? predictedMove : -predictedMove,
        absoluteMove: predictedMove,
        direction,
        isActionable: predictedMove >= 1.0,
        movePercent: (predictedMove / 75.5) * 100,
        threshold: 1.0,
        thresholdMultiple: predictedMove / 1.0,
      },
      horizonCoverage: {
        coveredHorizons,
        missingHorizons: ["D+7", "D+10"].filter(
          (h) => !coveredHorizons.includes(h as Horizon)
        ) as Horizon[],
        coveragePercent: (coveredHorizons.length / 6) * 100,
        hasShortTerm: coveredHorizons.some((h) => ["D+1", "D+2"].includes(h)),
        hasMediumTerm: coveredHorizons.some((h) => ["D+3", "D+5"].includes(h)),
        hasLongTerm: coveredHorizons.some((h) => ["D+7", "D+10"].includes(h)),
        optimalHorizon,
      },
      practicalScore: {
        score,
        actionability,
        components: {
          magnitudeScore: 85,
          horizonScore: 70,
          confidenceScore: 75,
          bigMoveAccuracyScore: 65,
        },
        weights: {
          magnitude: 0.35,
          horizon: 0.2,
          confidence: 0.2,
          bigMoveAccuracy: 0.25,
        },
      },
      confidence,
      bigMoveWinRate,
      recommendedPositionSize: positionSize,
      timeToAction: {
        urgency,
        daysToOptimalEntry: urgency === "immediate" ? 0 : 1,
        daysUntilExpiry: 5,
        reason:
          urgency === "immediate"
            ? "Strong signal - act now"
            : "Monitor for confirmation",
      },
    },
    traditionalSharpe,
    overallWinRate,
    analyzedAt: "2024-01-15T10:30:00Z",
  };
};

// ============================================================================
// Tests
// ============================================================================

describe("PracticalMetrics", () => {
  describe("loading state", () => {
    it("renders skeleton when isLoading is true", () => {
      render(<PracticalMetrics asset="crude-oil" isLoading={true} />);

      // Skeleton component uses animate-pulse class
      const skeletons = document.querySelectorAll('[class*="animate-pulse"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });

    it("does not render data content when loading", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          isLoading={true}
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.queryByText("Practical Metrics")).not.toBeInTheDocument();
    });
  });

  describe("error state", () => {
    it("renders error message when error is provided", () => {
      const error = new Error("Failed to fetch metrics");
      render(<PracticalMetrics asset="crude-oil" error={error} />);

      expect(screen.getByText("Failed to load practical metrics")).toBeInTheDocument();
      expect(screen.getByText("Failed to fetch metrics")).toBeInTheDocument();
    });

    it("has red border styling for error state", () => {
      const error = new Error("API error");
      render(<PracticalMetrics asset="crude-oil" error={error} />);

      const card = document.querySelector('[class*="border-red"]');
      expect(card).toBeInTheDocument();
    });
  });

  describe("empty state", () => {
    it("renders empty message when no data is provided", () => {
      render(<PracticalMetrics asset="crude-oil" />);

      expect(screen.getByText("No practical metrics available")).toBeInTheDocument();
    });
  });

  describe("actionability indicator", () => {
    it.each<[ActionabilityLevel, string]>([
      ["high", "High — Act Now"],
      ["medium", "Medium — Watch Closely"],
      ["low", "Low — Wait"],
    ])("displays correct label for %s actionability", (level, expectedLabel) => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ actionability: level })}
        />
      );

      expect(screen.getByText(expectedLabel)).toBeInTheDocument();
    });

    it("displays practical score as integer", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ score: 78.6 })}
        />
      );

      expect(screen.getByText("79")).toBeInTheDocument();
      expect(screen.getByText("Practical Score")).toBeInTheDocument();
    });

    it("renders pulsing dot indicator", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ actionability: "high" })}
        />
      );

      const dot = document.querySelector(".animate-pulse");
      expect(dot).toBeInTheDocument();
    });

    it("applies green styling for high actionability", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ actionability: "high" })}
        />
      );

      const greenElements = document.querySelectorAll('[class*="green"]');
      expect(greenElements.length).toBeGreaterThan(0);
    });

    it("applies yellow styling for medium actionability", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ actionability: "medium", score: 55 })}
        />
      );

      const indicator = screen.getByText("Medium — Watch Closely");
      expect(indicator.className).toContain("yellow");
    });

    it("applies red styling for low actionability", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ actionability: "low", score: 30 })}
        />
      );

      const indicator = screen.getByText("Low — Wait");
      expect(indicator.className).toContain("red");
    });
  });

  describe("urgency badge", () => {
    it.each<["immediate" | "today" | "this_week" | "monitor", string]>([
      ["immediate", "Act Now"],
      ["today", "Today"],
      ["this_week", "This Week"],
      ["monitor", "Monitor"],
    ])("displays correct text for %s urgency", (urgency, expectedText) => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ urgency })}
        />
      );

      expect(screen.getByText(expectedText)).toBeInTheDocument();
    });
  });

  describe("reasons section", () => {
    it("displays actionability reasons when actionable", () => {
      const reasons = ["Large predicted move", "High confidence"];
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({
            isActionable: true,
            reasons,
          })}
        />
      );

      expect(screen.getByText("Why Actionable")).toBeInTheDocument();
      reasons.forEach((reason) => {
        expect(screen.getByText(reason)).toBeInTheDocument();
      });
    });

    it("displays non-actionable reasons when not actionable", () => {
      const reasons = ["Move too small", "Low confidence"];
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({
            isActionable: false,
            actionability: "low",
            reasons,
          })}
        />
      );

      expect(screen.getByText("Why Not Actionable")).toBeInTheDocument();
    });

    it("renders green dots for actionable reasons", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ isActionable: true })}
        />
      );

      const greenDots = document.querySelectorAll(".bg-green-500");
      expect(greenDots.length).toBeGreaterThan(0);
    });
  });

  describe("forecast size display", () => {
    it("displays formatted move size for crude oil", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({
            asset: "crude-oil",
            predictedMove: 2.5,
            direction: "up",
          })}
        />
      );

      expect(screen.getByText("+$2.50")).toBeInTheDocument();
    });

    it("displays green color when magnitude is actionable", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ predictedMove: 2.5 })}
        />
      );

      const forecastValue = screen.getByText("+$2.50");
      expect(forecastValue.className).toContain("green");
    });

    it("displays threshold comparison as subtext", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText(/Target: >\+\$1\.00/)).toBeInTheDocument();
    });
  });

  describe("big move win rate", () => {
    it("displays win rate as percentage", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ bigMoveWinRate: 65.5 })}
        />
      );

      expect(screen.getByText("65.5%")).toBeInTheDocument();
    });

    it("applies green color for win rate >= 60", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ bigMoveWinRate: 65 })}
        />
      );

      const winRateElement = screen.getByText("65.0%");
      expect(winRateElement.className).toContain("green");
    });

    it("applies yellow color for win rate 50-59", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ bigMoveWinRate: 55 })}
        />
      );

      const winRateElement = screen.getByText("55.0%");
      expect(winRateElement.className).toContain("yellow");
    });

    it("applies red color for win rate < 50", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ bigMoveWinRate: 45 })}
        />
      );

      const winRateElement = screen.getByText("45.0%");
      expect(winRateElement.className).toContain("red");
    });
  });

  describe("position size recommendation", () => {
    it("displays position size as percentage", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ positionSize: 60 })}
        />
      );

      expect(screen.getByText("60%")).toBeInTheDocument();
      expect(screen.getByText("Of max exposure")).toBeInTheDocument();
    });

    it("applies green for position >= 50%", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ positionSize: 60 })}
        />
      );

      const positionElement = screen.getByText("60%");
      expect(positionElement.className).toContain("green");
    });

    it("applies yellow for position 25-49%", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ positionSize: 40 })}
        />
      );

      const positionElement = screen.getByText("40%");
      expect(positionElement.className).toContain("yellow");
    });

    it("applies neutral color for position < 25%", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ positionSize: 15 })}
        />
      );

      const positionElement = screen.getByText("15%");
      expect(positionElement.className).toContain("neutral");
    });
  });

  describe("model confidence", () => {
    it("displays confidence as integer percentage", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ confidence: 75.8 })}
        />
      );

      expect(screen.getByText("76%")).toBeInTheDocument();
    });

    it("applies correct color based on confidence level", () => {
      const { rerender } = render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ confidence: 75 })}
        />
      );

      // High confidence - green
      let confidenceElement = screen.getByText("75%");
      expect(confidenceElement.className).toContain("green");

      rerender(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ confidence: 55 })}
        />
      );

      // Medium confidence - yellow
      confidenceElement = screen.getByText("55%");
      expect(confidenceElement.className).toContain("yellow");

      rerender(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ confidence: 40 })}
        />
      );

      // Low confidence - red
      confidenceElement = screen.getByText("40%");
      expect(confidenceElement.className).toContain("red");
    });
  });

  describe("traditional sharpe ratio", () => {
    it("displays sharpe ratio with 2 decimal places", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ traditionalSharpe: 1.85 })}
        />
      );

      expect(screen.getByText("1.85")).toBeInTheDocument();
    });

    it("applies green for sharpe >= 1.5", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ traditionalSharpe: 1.8 })}
        />
      );

      const sharpeElement = screen.getByText("1.80");
      expect(sharpeElement.className).toContain("green");
    });

    it("applies yellow for sharpe 0.5-1.49", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ traditionalSharpe: 0.8 })}
        />
      );

      const sharpeElement = screen.getByText("0.80");
      expect(sharpeElement.className).toContain("yellow");
    });

    it("applies red for sharpe < 0.5", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ traditionalSharpe: 0.3 })}
        />
      );

      const sharpeElement = screen.getByText("0.30");
      expect(sharpeElement.className).toContain("red");
    });
  });

  describe("horizon coverage", () => {
    it("displays coverage percentage", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({
            coveredHorizons: ["D+1", "D+2", "D+3", "D+5"],
          })}
        />
      );

      expect(screen.getByText("67% covered")).toBeInTheDocument();
    });

    it("renders all 6 horizon pills", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText("D+1")).toBeInTheDocument();
      expect(screen.getByText("D+2")).toBeInTheDocument();
      expect(screen.getByText(/D\+3/)).toBeInTheDocument();
      expect(screen.getByText("D+5")).toBeInTheDocument();
      expect(screen.getByText("D+7")).toBeInTheDocument();
      expect(screen.getByText("D+10")).toBeInTheDocument();
    });

    it("marks optimal horizon with star", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ optimalHorizon: "D+3" })}
        />
      );

      expect(screen.getByText("D+3 ★")).toBeInTheDocument();
    });

    it("shows short-term coverage indicator", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({
            coveredHorizons: ["D+1", "D+2"],
          })}
        />
      );

      expect(screen.getByText(/✓.*Short-term/)).toBeInTheDocument();
    });

    it("shows missing coverage with ✗", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({
            coveredHorizons: ["D+1", "D+2"],
          })}
        />
      );

      expect(screen.getByText(/✗.*Long-term/)).toBeInTheDocument();
    });
  });

  describe("score breakdown", () => {
    it("displays all score components with weights", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText("Magnitude")).toBeInTheDocument();
      expect(screen.getByText("35%")).toBeInTheDocument();
      expect(screen.getByText("Horizons")).toBeInTheDocument();
      // 20% appears twice (for Horizons and Confidence weights)
      expect(screen.getAllByText("20%")).toHaveLength(2);
      expect(screen.getByText("Confidence")).toBeInTheDocument();
      expect(screen.getByText("Big Move Win")).toBeInTheDocument();
      expect(screen.getByText("25%")).toBeInTheDocument();
    });

    it("renders progress bars for each component", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData()}
        />
      );

      // Progress bars should have width styles
      const progressBars = document.querySelectorAll('[style*="width"]');
      expect(progressBars.length).toBeGreaterThan(0);
    });
  });

  describe("time to action", () => {
    it("displays time to action reason", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ urgency: "immediate" })}
        />
      );

      expect(screen.getByText("Strong signal - act now")).toBeInTheDocument();
    });

    it("displays days to optimal entry when > 0", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ urgency: "today" })}
        />
      );

      expect(screen.getByText(/Optimal entry in/)).toBeInTheDocument();
      expect(screen.getByText("1d")).toBeInTheDocument();
    });

    it("displays expiry countdown", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText(/Expires in/)).toBeInTheDocument();
      expect(screen.getByText("5d")).toBeInTheDocument();
    });
  });

  describe("footer", () => {
    it("displays utility level text for high actionability", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ actionability: "high" })}
        />
      );

      expect(screen.getByText("Strong practical utility")).toBeInTheDocument();
    });

    it("displays utility level text for medium actionability", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ actionability: "medium" })}
        />
      );

      expect(screen.getByText("Moderate practical utility")).toBeInTheDocument();
    });

    it("displays utility level text for low actionability", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ actionability: "low" })}
        />
      );

      expect(screen.getByText("Limited practical utility")).toBeInTheDocument();
    });

    it("displays analyzed timestamp", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData()}
        />
      );

      // Timestamp should be formatted as locale string
      const timestampElement = screen.getByText(/2024/);
      expect(timestampElement).toBeInTheDocument();
    });
  });

  describe("header badges", () => {
    it("displays Practical Metrics title", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText("Practical Metrics")).toBeInTheDocument();
    });

    it("displays CME Utility badge", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText("CME Utility")).toBeInTheDocument();
    });
  });

  describe("different assets", () => {
    it("formats bitcoin move size correctly", () => {
      const data = createMockPracticalMetricsData({
        asset: "bitcoin",
        predictedMove: 1500,
      });
      // Override the asset in the data
      data.asset = "bitcoin";
      data.currentPrice = 45000;
      data.signal.magnitude.predictedMove = 1500;
      data.signal.magnitude.absoluteMove = 1500;
      data.signal.magnitude.threshold = 500;

      render(<PracticalMetrics asset="bitcoin" data={data} />);

      expect(screen.getByText("+$1,500")).toBeInTheDocument();
    });

    it("formats agricultural asset move size with cents", () => {
      const data = createMockPracticalMetricsData({
        asset: "wheat",
        predictedMove: 8,
      });
      data.asset = "wheat";
      data.signal.magnitude.predictedMove = 8;
      data.signal.magnitude.absoluteMove = 8;
      data.signal.magnitude.threshold = 5;

      render(<PracticalMetrics asset="wheat" data={data} />);

      expect(screen.getByText("+8.0¢")).toBeInTheDocument();
    });
  });

  describe("edge cases", () => {
    it("handles zero score", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ score: 0 })}
        />
      );

      expect(screen.getByText("0")).toBeInTheDocument();
    });

    it("handles max score of 100", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ score: 100 })}
        />
      );

      expect(screen.getByText("100")).toBeInTheDocument();
    });

    it("handles empty reasons array", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ reasons: [] })}
        />
      );

      // Should still render without errors
      expect(screen.getByText("Practical Metrics")).toBeInTheDocument();
    });

    it("handles null optimal horizon", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ optimalHorizon: null })}
        />
      );

      // No star should be displayed
      expect(screen.queryByText(/★/)).not.toBeInTheDocument();
    });

    it("handles bearish direction", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ direction: "down", predictedMove: 2.5 })}
        />
      );

      expect(screen.getByText("-$2.50")).toBeInTheDocument();
    });

    it("handles very small move sizes", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({ predictedMove: 0.1 })}
        />
      );

      expect(screen.getByText("+$0.10")).toBeInTheDocument();
    });

    it("handles 100% horizon coverage", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({
            coveredHorizons: ["D+1", "D+2", "D+3", "D+5", "D+7", "D+10"],
          })}
        />
      );

      expect(screen.getByText("100% covered")).toBeInTheDocument();
    });

    it("handles 0% horizon coverage", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData({
            coveredHorizons: [],
          })}
        />
      );

      expect(screen.getByText("0% covered")).toBeInTheDocument();
    });
  });

  describe("responsive layout", () => {
    it("renders grid with correct column classes", () => {
      render(
        <PracticalMetrics
          asset="crude-oil"
          data={createMockPracticalMetricsData()}
        />
      );

      const grid = document.querySelector(".grid.grid-cols-2.md\\:grid-cols-3");
      expect(grid).toBeInTheDocument();
    });
  });
});
