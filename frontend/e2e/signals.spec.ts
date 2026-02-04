import { test, expect } from '@playwright/test';

test.describe('Signal Cards', () => {
  test('should render signal cards', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Wait for signal cards to load
    const signalCards = page.locator('[data-testid="signal-card"]');
    await expect(signalCards.first()).toBeVisible({ timeout: 15000 });

    // Verify multiple cards are rendered
    const count = await signalCards.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should display correct direction attributes', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Wait for signal cards to load
    await page.locator('[data-testid="signal-card"]').first().waitFor({ timeout: 15000 });

    // Check for bullish signals
    const bullishSignals = page.locator('[data-testid="signal-card"][data-direction="bullish"]');
    const bearishSignals = page.locator('[data-testid="signal-card"][data-direction="bearish"]');
    const neutralSignals = page.locator('[data-testid="signal-card"][data-direction="neutral"]');

    // Count each type
    const bullishCount = await bullishSignals.count();
    const bearishCount = await bearishSignals.count();
    const neutralCount = await neutralSignals.count();

    // At least one signal type should exist
    const totalSignals = bullishCount + bearishCount + neutralCount;
    expect(totalSignals).toBeGreaterThan(0);

    // Verify data-direction attributes are valid
    if (bullishCount > 0) {
      await expect(bullishSignals.first()).toHaveAttribute('data-direction', 'bullish');
    }
    if (bearishCount > 0) {
      await expect(bearishSignals.first()).toHaveAttribute('data-direction', 'bearish');
    }
  });

  test('should navigate to asset detail on click', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Wait for signal cards to load
    const signalCard = page.locator('[data-testid="signal-card"]').first();
    await expect(signalCard).toBeVisible({ timeout: 15000 });

    // Click and wait for navigation
    await signalCard.click();
    
    // Wait for URL to change
    await page.waitForURL(/\/dashboard\/assets\//, { timeout: 15000 });
    
    // Verify navigation
    await expect(page).toHaveURL(/\/dashboard\/assets\//);
  });
});
