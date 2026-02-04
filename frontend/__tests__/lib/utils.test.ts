/**
 * Tests for lib/utils.ts
 */

import { cn } from "@/lib/utils";

describe("cn (className merge utility)", () => {
  it("merges simple class names", () => {
    expect(cn("foo", "bar")).toBe("foo bar");
  });

  it("handles undefined and null values", () => {
    expect(cn("foo", undefined, "bar", null)).toBe("foo bar");
  });

  it("handles empty strings", () => {
    expect(cn("foo", "", "bar")).toBe("foo bar");
  });

  it("handles conditional classes", () => {
    const isActive = true;
    const isDisabled = false;

    expect(cn("base", isActive && "active", isDisabled && "disabled")).toBe(
      "base active"
    );
  });

  it("merges Tailwind classes correctly", () => {
    // tailwind-merge should handle conflicting utilities
    expect(cn("px-4", "px-6")).toBe("px-6");
    expect(cn("text-red-500", "text-blue-500")).toBe("text-blue-500");
  });

  it("handles object syntax from clsx", () => {
    expect(
      cn({
        "text-red-500": true,
        "bg-blue-500": false,
        "font-bold": true,
      })
    ).toBe("text-red-500 font-bold");
  });

  it("handles array syntax", () => {
    expect(cn(["foo", "bar"], "baz")).toBe("foo bar baz");
  });

  it("handles complex combinations", () => {
    const result = cn(
      "base-class",
      { conditional: true, "not-included": false },
      ["array-class"],
      undefined,
      "final-class"
    );

    expect(result).toContain("base-class");
    expect(result).toContain("conditional");
    expect(result).not.toContain("not-included");
    expect(result).toContain("array-class");
    expect(result).toContain("final-class");
  });

  it("handles responsive modifiers", () => {
    expect(cn("sm:text-lg", "md:text-xl", "lg:text-2xl")).toBe(
      "sm:text-lg md:text-xl lg:text-2xl"
    );
  });

  it("handles pseudo-class modifiers", () => {
    expect(cn("hover:bg-blue-500", "focus:ring-2")).toBe(
      "hover:bg-blue-500 focus:ring-2"
    );
  });

  it("returns empty string for no valid classes", () => {
    expect(cn(undefined, null, false, "")).toBe("");
  });
});
