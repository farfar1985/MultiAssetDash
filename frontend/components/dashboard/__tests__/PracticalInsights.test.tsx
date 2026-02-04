import { render, screen } from "@testing-library/react";
import { PracticalInsights } from "../PracticalInsights";
import type { PracticalMetricsData, ActionabilityLevel } from "@/types/practical-metrics";
import type { AssetId } from "@/types";
import type { Horizon } from "@/types/horizon-pairs";

// ============================================================================
// Test Fixtures
// ============================================================================

const createMockPracticalMetricsData = (
  overrides: Partial<{
    asset: AssetId;
    assetName: string;
    actionability: ActionabilityLevel;
    score: number;
    isActionable: boolean;
    direction: "up" | "down";
    predictedMove: number;
    absoluteMove: number;
    movePercent: number;
    thresholdMultiple: number;
    confidence: number;
    bigMoveWinRate: number;
    positionSize: number;
    coveredHorizons: Horizon[];
    optimalHorizon: Horizon | null;
    traditionalSharpe: number;
    overallWinRate: number;
    currentPrice: number;
  }> = {}
): PracticalMetricsData => {
  const {
    asset = "crude-oil",
    actionability = "high",
    score = 78,
    isActionable = true,
    direction = "up",
    predictedMove = 2.5,
    absoluteMove = 2.5,
    movePercent = 3.31,
    thresholdMultiple = 2.5,
    confidence = 75,
    bigMoveWinRate = 65,
    positionSize = 60,
    coveredHorizons = ["D+1", "D+2", "D+3", "D+5"],
    optimalHorizon = "D+3",
    traditionalSharpe = 1.8,
    overallWinRate = 58,
    currentPrice = 75.5,
  } = overrides;

  return {
    asset,
    currentPrice,
    signal: {
      isActionable,
      reasons: ["Large predicted move", "High confidence"],
      magnitude: {
        predictedMove: direction === "up" ? predictedMove : -predictedMove,
        absoluteMove,
        direction,
        isActionable: absoluteMove >= 1.0,
        movePercent,
        threshold: 1.0,
        thresholdMultiple,
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
        urgency: "immediate",
        daysToOptimalEntry: 0,
        daysUntilExpiry: 5,
        reason: "Strong signal - act now",
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

describe("PracticalInsights", () => {
  describe("loading state", () => {
    it("renders skeleton when isLoading is true", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          isLoading={true}
        />
      );

      // Skeleton component uses animate-pulse class
      const skeletons = document.querySelectorAll('[class*="animate-pulse"]');
      expect(skeletons.length).toBeGreaterThan(0);
    });

    it("does not render insights when loading", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          isLoading={true}
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.queryByText("Trading Insights")).not.toBeInTheDocument();
    });
  });

  describe("error state", () => {
    it("renders error message when error is provided", () => {
      const error = new Error("API Error");
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          error={error}
        />
      );

      expect(screen.getByText("Failed to load insights")).toBeInTheDocument();
      expect(screen.getByText("API Error")).toBeInTheDocument();
    });

    it("applies red border styling for error state", () => {
      const error = new Error("Error");
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          error={error}
        />
      );

      const card = document.querySelector('[class*="border-red"]');
      expect(card).toBeInTheDocument();
    });
  });

  describe("empty state", () => {
    it("renders empty message when no data is provided", () => {
      render(
        <PracticalInsights asset="crude-oil" assetName="Crude Oil" />
      );

      expect(screen.getByText("No insights available")).toBeInTheDocument();
    });
  });

  describe("header", () => {
    it("displays Trading Insights title", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText("Trading Insights")).toBeInTheDocument();
    });

    it("displays Plain English badge", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText("Plain English")).toBeInTheDocument();
    });
  });

  describe("confidence badge", () => {
    it("displays High Confidence badge for high confidence insight", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            confidence: 75,
            bigMoveWinRate: 65,
            absoluteMove: 2.5,
          })}
        />
      );

      expect(screen.getByText("High Confidence")).toBeInTheDocument();
    });

    it("displays Medium Confidence badge for medium confidence insight", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            confidence: 55,
            bigMoveWinRate: 55,
          })}
        />
      );

      expect(screen.getByText("Medium Confidence")).toBeInTheDocument();
    });

    it("displays Low Confidence badge for low confidence insight", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            confidence: 40,
            bigMoveWinRate: 45,
          })}
        />
      );

      expect(screen.getByText("Low Confidence")).toBeInTheDocument();
    });

    it("applies green styling for high confidence", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            confidence: 75,
            bigMoveWinRate: 65,
          })}
        />
      );

      const badge = screen.getByText("High Confidence");
      expect(badge.className).toContain("green");
    });

    it("applies yellow styling for medium confidence", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            confidence: 55,
            bigMoveWinRate: 55,
          })}
        />
      );

      const badge = screen.getByText("Medium Confidence");
      expect(badge.className).toContain("yellow");
    });
  });

  describe("headline generation", () => {
    it("generates bullish headline for upward direction", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            direction: "up",
            absoluteMove: 2.5,
            optimalHorizon: "D+3",
          })}
        />
      );

      expect(screen.getByText(/Crude Oil:.*Bullish signal/i)).toBeInTheDocument();
    });

    it("generates bearish headline for downward direction", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            direction: "down",
            absoluteMove: 2.5,
            optimalHorizon: "D+3",
          })}
        />
      );

      expect(screen.getByText(/Crude Oil:.*Bearish signal/i)).toBeInTheDocument();
    });

    it("includes move size in headline", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            absoluteMove: 2.5,
          })}
        />
      );

      expect(screen.getByText(/\$2\.50 move/i)).toBeInTheDocument();
    });

    it("includes horizon days in headline", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            optimalHorizon: "D+5",
          })}
        />
      );

      expect(screen.getByText(/5 days/i)).toBeInTheDocument();
    });

    it("applies green border for high actionability", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({ actionability: "high" })}
        />
      );

      const headlineBox = document.querySelector(".border-l-green-500");
      expect(headlineBox).toBeInTheDocument();
    });

    it("applies yellow border for medium actionability", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({ actionability: "medium" })}
        />
      );

      const headlineBox = document.querySelector(".border-l-yellow-500");
      expect(headlineBox).toBeInTheDocument();
    });

    it("applies neutral border for low actionability", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({ actionability: "low" })}
        />
      );

      const headlineBox = document.querySelector(".border-l-neutral-600");
      expect(headlineBox).toBeInTheDocument();
    });
  });

  describe("quick stats row", () => {
    it("displays move stat with correct value", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            direction: "up",
            predictedMove: 2.5,
          })}
        />
      );

      expect(screen.getByText("+$2.50")).toBeInTheDocument();
      expect(screen.getByText("Move")).toBeInTheDocument();
    });

    it("displays horizon stat", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            optimalHorizon: "D+3",
          })}
        />
      );

      expect(screen.getByText("D+3")).toBeInTheDocument();
      expect(screen.getByText("Horizon")).toBeInTheDocument();
    });

    it("displays — when no optimal horizon", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            optimalHorizon: null,
          })}
        />
      );

      expect(screen.getByText("—")).toBeInTheDocument();
    });

    it("displays confidence stat", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            confidence: 75,
          })}
        />
      );

      expect(screen.getByText("75%")).toBeInTheDocument();
      // Multiple "Confidence" texts may appear - one for stat label, one for badge
    });

    it("displays position stat", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            positionSize: 60,
          })}
        />
      );

      expect(screen.getByText("60%")).toBeInTheDocument();
      expect(screen.getByText("Position")).toBeInTheDocument();
    });

    it("applies green color for bullish move", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            direction: "up",
          })}
        />
      );

      const moveValue = screen.getByText("+$2.50");
      expect(moveValue.className).toContain("green");
    });

    it("applies red color for bearish move", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            direction: "down",
            predictedMove: 2.5,
          })}
        />
      );

      const moveValue = screen.getByText("-$2.50");
      expect(moveValue.className).toContain("red");
    });
  });

  describe("explanation section", () => {
    it("displays What This Means section", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText("What This Means")).toBeInTheDocument();
    });

    it("includes direction in explanation", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            direction: "up",
          })}
        />
      );

      expect(screen.getByText(/positive move/i)).toBeInTheDocument();
    });

    it("includes move percentage in explanation", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            movePercent: 3.31,
          })}
        />
      );

      expect(screen.getByText(/3\.31%/)).toBeInTheDocument();
    });

    it("mentions threshold when actionable", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            isActionable: true,
            thresholdMultiple: 2.5,
          })}
        />
      );

      expect(screen.getByText(/exceeds.*threshold.*2\.5x/i)).toBeInTheDocument();
    });

    it("mentions below threshold when not actionable", () => {
      const data = createMockPracticalMetricsData({
        actionability: "low",
      });
      data.signal.magnitude.isActionable = false;

      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={data}
        />
      );

      // "below" and "threshold" appear in multiple places - check explanation section
      // The explanation mentions "below the ... threshold needed for actionability"
      expect(screen.getByText(/This is below the.*threshold needed for actionability/i)).toBeInTheDocument();
    });

    it("includes confidence and win rate in explanation", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            confidence: 75,
            bigMoveWinRate: 65,
          })}
        />
      );

      expect(screen.getByText(/75%.*65%/)).toBeInTheDocument();
    });
  });

  describe("recommendation section", () => {
    it("displays Recommendation section", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText("Recommendation")).toBeInTheDocument();
    });

    it("shows hedging recommendation for high actionability bullish", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            actionability: "high",
            direction: "up",
            positionSize: 60,
          })}
        />
      );

      expect(screen.getByText(/hedging.*60%.*short exposure/i)).toBeInTheDocument();
    });

    it("shows hedging recommendation for high actionability bearish", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            actionability: "high",
            direction: "down",
            positionSize: 60,
          })}
        />
      );

      expect(screen.getByText(/hedging.*60%.*long exposure/i)).toBeInTheDocument();
    });

    it("shows monitor recommendation for medium actionability", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            actionability: "medium",
            positionSize: 60,
          })}
        />
      );

      expect(screen.getByText(/Monitor closely.*30% position/i)).toBeInTheDocument();
    });

    it("shows no action recommendation for low actionability", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            actionability: "low",
          })}
        />
      );

      expect(screen.getByText(/No action recommended/i)).toBeInTheDocument();
    });

    it("applies highlight variant for high actionability", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({ actionability: "high" })}
        />
      );

      const recommendationSection = document.querySelector(".bg-green-500\\/5");
      expect(recommendationSection).toBeInTheDocument();
    });

    it("applies warning variant for medium actionability", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({ actionability: "medium" })}
        />
      );

      const warningSection = document.querySelector(".bg-yellow-500\\/5");
      expect(warningSection).toBeInTheDocument();
    });
  });

  describe("actionability reason section", () => {
    it("displays Why This Rating section", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText("Why This Rating")).toBeInTheDocument();
    });

    it("includes large forecast reason when actionable", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            isActionable: true,
          })}
        />
      );

      expect(screen.getByText(/large forecast size/i)).toBeInTheDocument();
    });

    it("includes horizon diversity info in reasons", () => {
      // 5 out of 6 horizons = 83% coverage >= 67%
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            coveredHorizons: ["D+1", "D+2", "D+3", "D+5", "D+7"],
          })}
        />
      );

      // For actionable signals with good coverage, reason includes "good horizon diversity"
      expect(screen.getByText(/good horizon diversity/i)).toBeInTheDocument();
    });

    it("includes high confidence reason when >= 70", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            confidence: 75,
          })}
        />
      );

      // High confidence is included in the actionability reason
      expect(screen.getByText(/high model confidence/i)).toBeInTheDocument();
    });

    it("shows 'Actionable because' for actionable signals", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            confidence: 75,
            bigMoveWinRate: 65,
          })}
        />
      );

      // The actionability reason should start with "Actionable because:"
      expect(screen.getByText(/Actionable because:/i)).toBeInTheDocument();
      // And include key reasons (only first 3 are shown)
      expect(screen.getByText(/large forecast size/i)).toBeInTheDocument();
    });

    it("mentions below threshold when not actionable", () => {
      const data = createMockPracticalMetricsData({
        actionability: "low",
      });
      data.signal.magnitude.isActionable = false;
      data.signal.isActionable = false;

      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={data}
        />
      );

      expect(screen.getByText(/Not actionable.*forecast below threshold/i)).toBeInTheDocument();
    });
  });

  describe("risk note section", () => {
    it("displays Risk Considerations section", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText("Risk Considerations")).toBeInTheDocument();
    });

    it("shows appropriate risk note for high actionability", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            actionability: "high",
          })}
        />
      );

      expect(screen.getByText(/Even strong signals can be wrong/i)).toBeInTheDocument();
    });

    it("shows appropriate risk note for medium actionability", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            actionability: "medium",
            bigMoveWinRate: 55,
          })}
        />
      );

      expect(screen.getByText(/Medium signals have.*55%.*success rate/i)).toBeInTheDocument();
    });

    it("shows appropriate risk note for low actionability", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            actionability: "low",
          })}
        />
      );

      expect(screen.getByText(/Weak signals are often noise/i)).toBeInTheDocument();
    });
  });

  describe("traditional vs practical comparison", () => {
    it("displays Traditional vs. Practical Comparison section", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText("Traditional vs. Practical Comparison")).toBeInTheDocument();
    });

    it("displays traditional sharpe value", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            traditionalSharpe: 1.85,
          })}
        />
      );

      expect(screen.getByText("1.85")).toBeInTheDocument();
    });

    it("displays practical score value", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            score: 78,
          })}
        />
      );

      expect(screen.getByText("78")).toBeInTheDocument();
    });

    it("shows 'Looks great on paper' for high sharpe >= 1.5", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            traditionalSharpe: 1.8,
          })}
        />
      );

      expect(screen.getByText("Looks great on paper")).toBeInTheDocument();
    });

    it("shows 'Acceptable risk-adjusted' for sharpe 0.5-1.49", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            traditionalSharpe: 1.0,
          })}
        />
      );

      expect(screen.getByText("Acceptable risk-adjusted")).toBeInTheDocument();
    });

    it("shows 'Poor risk-adjusted' for sharpe < 0.5", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            traditionalSharpe: 0.3,
          })}
        />
      );

      expect(screen.getByText("Poor risk-adjusted")).toBeInTheDocument();
    });

    it("shows 'Actually tradeable' for high practical score", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            actionability: "high",
          })}
        />
      );

      expect(screen.getByText("Actually tradeable")).toBeInTheDocument();
    });

    it("shows 'Marginally useful' for medium practical score", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            actionability: "medium",
          })}
        />
      );

      expect(screen.getByText("Marginally useful")).toBeInTheDocument();
    });

    it("shows 'Not worth the effort' for low practical score", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            actionability: "low",
          })}
        />
      );

      expect(screen.getByText("Not worth the effort")).toBeInTheDocument();
    });

    it("shows warning when high sharpe but low practical score", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            traditionalSharpe: 1.8,
            actionability: "low",
          })}
        />
      );

      expect(
        screen.getByText(/High Sharpe but low practical score/i)
      ).toBeInTheDocument();
      expect(screen.getByText(/moves are too small to action/i)).toBeInTheDocument();
    });

    it("shows positive note when moderate sharpe but high practical score", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            traditionalSharpe: 0.8,
            actionability: "high",
          })}
        />
      );

      expect(
        screen.getByText(/Moderate Sharpe but high practical score/i)
      ).toBeInTheDocument();
    });

    it("does not show special message when sharpe and practical align", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            traditionalSharpe: 1.8,
            actionability: "high",
          })}
        />
      );

      expect(
        screen.queryByText(/High Sharpe but low practical score/i)
      ).not.toBeInTheDocument();
      expect(
        screen.queryByText(/Moderate Sharpe but high practical score/i)
      ).not.toBeInTheDocument();
    });
  });

  describe("footer", () => {
    it("displays CME hedging desk utility note", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(
        screen.getByText("Insights generated for CME hedging desk utility")
      ).toBeInTheDocument();
    });

    it("displays timestamp", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData()}
        />
      );

      const timestampElement = screen.getByText(/2024/);
      expect(timestampElement).toBeInTheDocument();
    });
  });

  describe("insight section variants", () => {
    it("renders default variant sections", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData()}
        />
      );

      // What This Means should have default styling
      const defaultSection = document.querySelector(".bg-neutral-800\\/30");
      expect(defaultSection).toBeInTheDocument();
    });

    it("renders muted variant for risk section", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData()}
        />
      );

      const mutedSection = document.querySelector(".bg-neutral-800\\/20");
      expect(mutedSection).toBeInTheDocument();
    });

    it("displays icons for each section", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData()}
        />
      );

      expect(screen.getByText("📊")).toBeInTheDocument(); // What This Means
      expect(screen.getByText("💡")).toBeInTheDocument(); // Recommendation
      expect(screen.getByText("🎯")).toBeInTheDocument(); // Why This Rating
      expect(screen.getByText("⚠️")).toBeInTheDocument(); // Risk
    });
  });

  describe("edge cases", () => {
    it("handles different asset types", () => {
      const data = createMockPracticalMetricsData({
        asset: "bitcoin",
        absoluteMove: 1500,
        predictedMove: 1500,
        direction: "up",
      });
      data.asset = "bitcoin";
      data.currentPrice = 45000;
      data.signal.magnitude.threshold = 500;

      render(
        <PracticalInsights asset="bitcoin" assetName="Bitcoin" data={data} />
      );

      expect(screen.getByText(/Bitcoin:.*Bullish/i)).toBeInTheDocument();
    });

    it("handles very low confidence", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            confidence: 30,
            bigMoveWinRate: 40,
          })}
        />
      );

      // Badge should show Low Confidence
      expect(screen.getByText("Low Confidence")).toBeInTheDocument();
      // The actionability reason should include low confidence
      // Multiple elements contain "low confidence" so use getAllByText
      const lowConfidenceElements = screen.getAllByText(/low confidence/i);
      expect(lowConfidenceElements.length).toBeGreaterThanOrEqual(1);
    });

    it("handles limited horizon coverage", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            coveredHorizons: ["D+1"],
          })}
        />
      );

      // For actionable signals with limited coverage, it should appear in the reasons
      expect(screen.getByText(/limited horizon coverage/i)).toBeInTheDocument();
    });

    it("handles poor big move accuracy in non-actionable signal", () => {
      // Poor big-move accuracy only shows for non-actionable signals
      // (actionable signals only show first 3 reasons which don't include it)
      const data = createMockPracticalMetricsData({
        bigMoveWinRate: 45,
        actionability: "low",
      });
      data.signal.isActionable = false;
      data.signal.magnitude.isActionable = false;

      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={data}
        />
      );

      // For non-actionable signals, poor big-move accuracy is included
      expect(screen.getByText(/Not actionable:.*poor big-move accuracy/i)).toBeInTheDocument();
    });

    it("uses D+5 as default horizon when optimal is null", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            optimalHorizon: null,
          })}
        />
      );

      // The headline should default to D+5 = 5 days
      expect(screen.getByText(/5 days/i)).toBeInTheDocument();
    });

    it("handles zero position size recommendation", () => {
      render(
        <PracticalInsights
          asset="crude-oil"
          assetName="Crude Oil"
          data={createMockPracticalMetricsData({
            positionSize: 0,
            actionability: "low",
          })}
        />
      );

      expect(screen.getByText("0%")).toBeInTheDocument();
    });
  });
});
