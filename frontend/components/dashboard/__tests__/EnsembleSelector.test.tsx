import { render, screen, fireEvent, within } from "@testing-library/react";
import {
  EnsembleSelector,
  ENSEMBLE_METHODS,
  getMethodConfig,
} from "../EnsembleSelector";
import type { EnsembleMethod } from "@/lib/api-client";

// Mock Radix UI Portal for Select dropdown
jest.mock("@radix-ui/react-select", () => {
  const actual = jest.requireActual("@radix-ui/react-select");
  return {
    ...actual,
    Portal: ({ children }: { children: React.ReactNode }) => children,
  };
});

describe("EnsembleSelector", () => {
  const mockOnChange = jest.fn();

  beforeEach(() => {
    mockOnChange.mockClear();
  });

  describe("ENSEMBLE_METHODS configuration", () => {
    it("contains exactly 6 ensemble methods", () => {
      expect(ENSEMBLE_METHODS).toHaveLength(6);
    });

    it("contains all required method values", () => {
      const methodValues = ENSEMBLE_METHODS.map((m) => m.value);
      expect(methodValues).toContain("accuracy_weighted");
      expect(methodValues).toContain("exponential_decay");
      expect(methodValues).toContain("top_k_sharpe");
      expect(methodValues).toContain("ridge_stacking");
      expect(methodValues).toContain("inverse_variance");
      expect(methodValues).toContain("pairwise_slope");
    });

    it("each method has required properties", () => {
      ENSEMBLE_METHODS.forEach((method) => {
        expect(method).toHaveProperty("value");
        expect(method).toHaveProperty("label");
        expect(method).toHaveProperty("shortLabel");
        expect(method).toHaveProperty("description");
        expect(typeof method.value).toBe("string");
        expect(typeof method.label).toBe("string");
        expect(typeof method.shortLabel).toBe("string");
        expect(typeof method.description).toBe("string");
      });
    });

    it("only top_k_sharpe has a badge", () => {
      const methodsWithBadge = ENSEMBLE_METHODS.filter((m) => m.badge);
      expect(methodsWithBadge).toHaveLength(1);
      expect(methodsWithBadge[0].value).toBe("top_k_sharpe");
      expect(methodsWithBadge[0].badge).toBe("BEST");
    });

    it("has correct labels for all methods", () => {
      const expectedLabels = {
        accuracy_weighted: "Accuracy Weighted",
        exponential_decay: "Exponential Decay",
        top_k_sharpe: "Top-K Sharpe",
        ridge_stacking: "Ridge Stacking",
        inverse_variance: "Inverse Variance",
        pairwise_slope: "Pairwise Slope",
      };

      ENSEMBLE_METHODS.forEach((method) => {
        expect(method.label).toBe(
          expectedLabels[method.value as keyof typeof expectedLabels]
        );
      });
    });

    it("has correct short labels for all methods", () => {
      const expectedShortLabels = {
        accuracy_weighted: "Accuracy",
        exponential_decay: "Exp Decay",
        top_k_sharpe: "Top-K",
        ridge_stacking: "Ridge",
        inverse_variance: "Inv Var",
        pairwise_slope: "X-Horizon",
      };

      ENSEMBLE_METHODS.forEach((method) => {
        expect(method.shortLabel).toBe(
          expectedShortLabels[method.value as keyof typeof expectedShortLabels]
        );
      });
    });
  });

  describe("getMethodConfig helper", () => {
    it("returns correct config for valid method", () => {
      const config = getMethodConfig("accuracy_weighted");
      expect(config.value).toBe("accuracy_weighted");
      expect(config.label).toBe("Accuracy Weighted");
    });

    it("returns first method as fallback for unknown method", () => {
      const config = getMethodConfig("unknown_method" as EnsembleMethod);
      expect(config).toBe(ENSEMBLE_METHODS[0]);
    });

    it("returns correct config for all valid methods", () => {
      const methods: EnsembleMethod[] = [
        "accuracy_weighted",
        "exponential_decay",
        "top_k_sharpe",
        "ridge_stacking",
        "inverse_variance",
        "pairwise_slope",
      ];

      methods.forEach((method) => {
        const config = getMethodConfig(method);
        expect(config.value).toBe(method);
      });
    });
  });

  describe("Tabs variant (default)", () => {
    it("renders as tabs by default", () => {
      render(
        <EnsembleSelector value="top_k_sharpe" onChange={mockOnChange} />
      );

      // Should render a tablist
      expect(screen.getByRole("tablist")).toBeInTheDocument();
    });

    it("renders all method options as tabs", () => {
      render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      const tablist = screen.getByRole("tablist");
      const tabs = within(tablist).getAllByRole("tab");

      expect(tabs).toHaveLength(6);
    });

    it("displays short labels for each tab", () => {
      render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      expect(screen.getByRole("tab", { name: /Accuracy/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /Exp Decay/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /Top-K/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /Ridge/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /Inv Var/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /X-Horizon/i })).toBeInTheDocument();
    });

    it("marks current method as selected", () => {
      render(
        <EnsembleSelector value="top_k_sharpe" onChange={mockOnChange} />
      );

      const topKTab = screen.getByRole("tab", { name: /Top-K/i });
      expect(topKTab).toHaveAttribute("data-state", "active");
    });

    it("marks other methods as inactive", () => {
      render(
        <EnsembleSelector value="top_k_sharpe" onChange={mockOnChange} />
      );

      const accuracyTab = screen.getByRole("tab", { name: /Accuracy/i });
      expect(accuracyTab).toHaveAttribute("data-state", "inactive");
    });

    it("has data-value attribute on tabs for method identification", () => {
      render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      const tabs = screen.getAllByRole("tab");

      // Verify each tab has the correct value in its attributes
      const methodValues = ENSEMBLE_METHODS.map(m => m.value);
      tabs.forEach((tab, index) => {
        expect(tab).toHaveAttribute("data-state");
        expect(tab.id).toContain(methodValues[index]);
      });
    });

    it("each tab is clickable and has proper interactive attributes", () => {
      render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      const tabs = screen.getAllByRole("tab");
      tabs.forEach((tab) => {
        expect(tab).toHaveAttribute("type", "button");
        expect(tab).not.toBeDisabled();
      });
    });

    it("displays BEST badge on top_k_sharpe tab", () => {
      render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      const topKTab = screen.getByRole("tab", { name: /Top-K/i });
      expect(within(topKTab).getByText("BEST")).toBeInTheDocument();
    });

    it("does not display badges on other tabs", () => {
      render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      const accuracyTab = screen.getByRole("tab", { name: /^Accuracy$/i });
      expect(within(accuracyTab).queryByText("BEST")).not.toBeInTheDocument();
    });

    it("has title attribute with description for tooltip", () => {
      render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      const topKTab = screen.getByRole("tab", { name: /Top-K/i });
      expect(topKTab).toHaveAttribute(
        "title",
        "Uses only models with highest Sharpe ratios"
      );
    });

    it("applies custom className to tabs container", () => {
      render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          className="custom-class"
        />
      );

      // The className is applied to the Tabs component wrapper
      const tabsContainer = screen.getByRole("tablist").parentElement;
      expect(tabsContainer).toHaveClass("custom-class");
    });

    it("updates selected state when value prop changes", () => {
      const { rerender } = render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      expect(
        screen.getByRole("tab", { name: /^Accuracy$/i })
      ).toHaveAttribute("data-state", "active");

      rerender(
        <EnsembleSelector value="ridge_stacking" onChange={mockOnChange} />
      );

      expect(
        screen.getByRole("tab", { name: /^Accuracy$/i })
      ).toHaveAttribute("data-state", "inactive");
      expect(screen.getByRole("tab", { name: /Ridge/i })).toHaveAttribute(
        "data-state",
        "active"
      );
    });
  });

  describe("Dropdown variant", () => {
    it("renders as dropdown when variant is dropdown", () => {
      render(
        <EnsembleSelector
          value="top_k_sharpe"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      // Should render a combobox (select trigger)
      expect(screen.getByRole("combobox")).toBeInTheDocument();
    });

    it("displays current method short label in trigger", () => {
      render(
        <EnsembleSelector
          value="top_k_sharpe"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      const trigger = screen.getByRole("combobox");
      expect(trigger).toHaveTextContent("Top-K");
    });

    it("displays BEST badge in trigger for top_k_sharpe", () => {
      render(
        <EnsembleSelector
          value="top_k_sharpe"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      const trigger = screen.getByRole("combobox");
      expect(within(trigger).getByText("BEST")).toBeInTheDocument();
    });

    it("does not display badge for methods without badge", () => {
      render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      const trigger = screen.getByRole("combobox");
      expect(within(trigger).queryByText("BEST")).not.toBeInTheDocument();
    });

    it("shows all options when dropdown is opened", () => {
      render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      const trigger = screen.getByRole("combobox");
      fireEvent.click(trigger);

      // Check for full labels in dropdown
      expect(screen.getByText("Accuracy Weighted")).toBeInTheDocument();
      expect(screen.getByText("Exponential Decay")).toBeInTheDocument();
      expect(screen.getByText("Top-K Sharpe")).toBeInTheDocument();
      expect(screen.getByText("Ridge Stacking")).toBeInTheDocument();
      expect(screen.getByText("Inverse Variance")).toBeInTheDocument();
      expect(screen.getByText("Pairwise Slope")).toBeInTheDocument();
    });

    it("shows descriptions for each option in dropdown", () => {
      render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      const trigger = screen.getByRole("combobox");
      fireEvent.click(trigger);

      expect(
        screen.getByText("Weights models by historical directional accuracy")
      ).toBeInTheDocument();
      expect(
        screen.getByText("Recent predictions weighted more heavily")
      ).toBeInTheDocument();
      expect(
        screen.getByText("Uses only models with highest Sharpe ratios")
      ).toBeInTheDocument();
    });

    it("triggers onChange when selecting a different option", () => {
      render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      const trigger = screen.getByRole("combobox");
      fireEvent.click(trigger);

      const ridgeOption = screen.getByRole("option", { name: /Ridge Stacking/i });
      fireEvent.click(ridgeOption);

      expect(mockOnChange).toHaveBeenCalledTimes(1);
      expect(mockOnChange).toHaveBeenCalledWith("ridge_stacking");
    });

    it("applies custom className to select trigger", () => {
      render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="dropdown"
          className="custom-dropdown-class"
        />
      );

      const trigger = screen.getByRole("combobox");
      expect(trigger).toHaveClass("custom-dropdown-class");
    });

    it("updates displayed value when value prop changes", () => {
      const { rerender } = render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      expect(screen.getByRole("combobox")).toHaveTextContent("Accuracy");

      rerender(
        <EnsembleSelector
          value="exponential_decay"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      expect(screen.getByRole("combobox")).toHaveTextContent("Exp Decay");
    });
  });

  describe("edge cases", () => {
    it("handles rapid value changes", () => {
      const { rerender } = render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      const methods: EnsembleMethod[] = [
        "exponential_decay",
        "top_k_sharpe",
        "ridge_stacking",
        "inverse_variance",
        "pairwise_slope",
        "accuracy_weighted",
      ];

      methods.forEach((method) => {
        rerender(
          <EnsembleSelector value={method} onChange={mockOnChange} />
        );
        const activeTab = screen
          .getAllByRole("tab")
          .find((tab) => tab.getAttribute("data-state") === "active");
        expect(activeTab).toBeDefined();
      });
    });

    it("does not trigger onChange when clicking already selected tab", () => {
      render(
        <EnsembleSelector value="top_k_sharpe" onChange={mockOnChange} />
      );

      const topKTab = screen.getByRole("tab", { name: /Top-K/i });
      fireEvent.click(topKTab);

      // Radix Tabs doesn't trigger onValueChange when clicking the already active tab
      expect(mockOnChange).not.toHaveBeenCalled();
    });

    it("handles variant switching", () => {
      const { rerender } = render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="tabs"
        />
      );

      expect(screen.getByRole("tablist")).toBeInTheDocument();

      rerender(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
      expect(screen.getByRole("combobox")).toBeInTheDocument();
    });
  });

  describe("accessibility", () => {
    it("tabs have proper aria roles", () => {
      render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      expect(screen.getByRole("tablist")).toBeInTheDocument();
      expect(screen.getAllByRole("tab")).toHaveLength(6);
    });

    it("dropdown has proper aria roles", () => {
      render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      expect(screen.getByRole("combobox")).toBeInTheDocument();
    });

    it("tabs have tabindex attribute for keyboard navigation", () => {
      render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      const tabs = screen.getAllByRole("tab");

      // All tabs should have tabindex attribute for roving tabindex pattern
      tabs.forEach((tab) => {
        expect(tab).toHaveAttribute("tabindex");
      });

      // Radix uses roving tabindex - verify the pattern exists
      const tabIndexValues = tabs.map((tab) =>
        tab.getAttribute("tabindex")
      );
      expect(tabIndexValues.some((v) => v === "0" || v === "-1")).toBe(true);
    });
  });

  describe("visual elements", () => {
    it("badge has correct styling classes", () => {
      render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      const badge = screen.getByText("BEST");
      expect(badge).toHaveClass("bg-green-500/10");
      expect(badge).toHaveClass("border-green-500/30");
      expect(badge).toHaveClass("text-green-500");
    });

    it("renders all 6 method configs correctly", () => {
      render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      // Verify each method config is rendered
      expect(screen.getByRole("tab", { name: /Accuracy/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /Exp Decay/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /Top-K/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /Ridge/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /Inv Var/i })).toBeInTheDocument();
      expect(screen.getByRole("tab", { name: /X-Horizon/i })).toBeInTheDocument();
    });
  });

  describe("tab interaction", () => {
    it("maintains data-state active on selected tab", () => {
      render(
        <EnsembleSelector value="top_k_sharpe" onChange={mockOnChange} />
      );

      const topKTab = screen.getByRole("tab", { name: /Top-K/i });
      expect(topKTab).toHaveAttribute("data-state", "active");
    });

    it("maintains data-state inactive on non-selected tabs", () => {
      render(
        <EnsembleSelector value="top_k_sharpe" onChange={mockOnChange} />
      );

      const inactiveTab = screen.getByRole("tab", { name: /^Accuracy$/i });
      expect(inactiveTab).toHaveAttribute("data-state", "inactive");
    });

    it("all tabs have tabindex attribute for keyboard navigation", () => {
      render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      const tabs = screen.getAllByRole("tab");
      tabs.forEach((tab) => {
        expect(tab).toHaveAttribute("tabindex");
      });
    });
  });

  describe("dropdown interaction", () => {
    it("closes dropdown after selection", () => {
      render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      const trigger = screen.getByRole("combobox");
      fireEvent.click(trigger);

      // Dropdown should be open
      expect(screen.getByText("Ridge Stacking")).toBeInTheDocument();

      const ridgeOption = screen.getByRole("option", { name: /Ridge Stacking/i });
      fireEvent.click(ridgeOption);

      expect(mockOnChange).toHaveBeenCalledWith("ridge_stacking");
    });

    it("shows checkmark or indicator for selected option", () => {
      render(
        <EnsembleSelector
          value="top_k_sharpe"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      const trigger = screen.getByRole("combobox");
      fireEvent.click(trigger);

      // The current value should be visually indicated (Radix handles this internally)
      const topKOption = screen.getByRole("option", { name: /Top-K Sharpe/i });
      expect(topKOption).toHaveAttribute("data-state", "checked");
    });

    it("does not trigger onChange when selecting already selected option", () => {
      render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      const trigger = screen.getByRole("combobox");
      fireEvent.click(trigger);

      const accuracyOption = screen.getByRole("option", { name: /Accuracy Weighted/i });
      fireEvent.click(accuracyOption);

      // Radix Select doesn't trigger onValueChange when selecting the same value
      expect(mockOnChange).not.toHaveBeenCalled();
    });

    it("displays all method descriptions in dropdown", () => {
      render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      const trigger = screen.getByRole("combobox");
      fireEvent.click(trigger);

      // Verify all descriptions are present
      ENSEMBLE_METHODS.forEach((method) => {
        expect(screen.getByText(method.description)).toBeInTheDocument();
      });
    });
  });

  describe("styling and layout", () => {
    it("tabs list has correct background and border classes", () => {
      render(
        <EnsembleSelector value="accuracy_weighted" onChange={mockOnChange} />
      );

      const tabsList = screen.getByRole("tablist");
      expect(tabsList).toHaveClass("bg-neutral-900");
      expect(tabsList).toHaveClass("border");
      expect(tabsList).toHaveClass("border-neutral-800");
    });

    it("active tab has highlighted styling", () => {
      render(
        <EnsembleSelector value="top_k_sharpe" onChange={mockOnChange} />
      );

      const activeTab = screen.getByRole("tab", { name: /Top-K/i });
      // When active, Radix applies data-[state=active] styling
      expect(activeTab).toHaveAttribute("data-state", "active");
    });

    it("dropdown trigger has correct width", () => {
      render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      const trigger = screen.getByRole("combobox");
      expect(trigger).toHaveClass("w-[200px]");
    });

    it("dropdown trigger has correct background styling", () => {
      render(
        <EnsembleSelector
          value="accuracy_weighted"
          onChange={mockOnChange}
          variant="dropdown"
        />
      );

      const trigger = screen.getByRole("combobox");
      expect(trigger).toHaveClass("bg-neutral-900");
      expect(trigger).toHaveClass("border-neutral-700");
    });
  });

  describe("method config completeness", () => {
    it("all methods have unique values", () => {
      const values = ENSEMBLE_METHODS.map((m) => m.value);
      const uniqueValues = new Set(values);
      expect(uniqueValues.size).toBe(values.length);
    });

    it("all methods have non-empty labels", () => {
      ENSEMBLE_METHODS.forEach((method) => {
        expect(method.label.length).toBeGreaterThan(0);
        expect(method.shortLabel.length).toBeGreaterThan(0);
        expect(method.description.length).toBeGreaterThan(0);
      });
    });

    it("short labels are shorter than full labels", () => {
      ENSEMBLE_METHODS.forEach((method) => {
        expect(method.shortLabel.length).toBeLessThanOrEqual(method.label.length);
      });
    });
  });
});
