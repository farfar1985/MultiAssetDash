import { test, expect } from '@playwright/test';

test.describe('Dashboard', () => {
  test('should load the dashboard', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    await expect(page).toHaveTitle(/Nexus/i);
    await expect(page.locator('main')).toBeVisible();
  });

  test('should navigate to asset pages', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Click on a signal card (which navigates to asset detail)
    const signalCard = page.locator('[data-testid="signal-card"]').first();
    
    if (await signalCard.isVisible()) {
      await signalCard.click();
      await page.waitForURL(/\/dashboard\/assets\//, { timeout: 10000 });
      await expect(page).toHaveURL(/\/dashboard\/assets\//);
    }
  });

  test('should switch personas', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Find and click the persona selector if it exists
    const personaSelector = page.locator('[data-testid="persona-selector"]');
    if (await personaSelector.isVisible()) {
      await personaSelector.click();

      // Select a different persona
      const personaOption = page.locator('[data-testid="persona-option"]').first();
      await personaOption.click();

      // Verify persona changed
      await expect(personaSelector).not.toHaveText(/Select/);
    }
  });

  test('should open command palette with keyboard shortcut', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Press Ctrl+K to open command palette
    await page.keyboard.press('Control+k');

    // Command palette should be visible
    const commandPalette = page.locator('[data-testid="command-palette"]');
    await expect(commandPalette).toBeVisible();
  });
});
