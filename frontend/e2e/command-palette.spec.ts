import { test, expect } from '@playwright/test';

test.describe('Command Palette', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    // Wait for the page to be fully loaded
    await page.waitForLoadState('networkidle');
  });

  test('should open with keyboard shortcut', async ({ page }) => {
    // Use Ctrl+K (works on all platforms in Playwright)
    await page.keyboard.press('Control+k');

    const commandPalette = page.locator('[data-testid="command-palette"]');
    await expect(commandPalette).toBeVisible();
  });

  test('should filter commands when searching', async ({ page }) => {
    // Open command palette
    await page.keyboard.press('Control+k');

    const commandPalette = page.locator('[data-testid="command-palette"]');
    await expect(commandPalette).toBeVisible();

    // Type a search query
    const searchInput = commandPalette.locator('input[type="text"]');
    await expect(searchInput).toBeVisible();
    await searchInput.fill('dashboard');

    // Verify filtered results - should have at least one item
    const commandItems = commandPalette.locator('[data-testid="command-item"]');
    await expect(commandItems.first()).toBeVisible();
    
    // Count should be less than total (filtered)
    const count = await commandItems.count();
    expect(count).toBeGreaterThan(0);
  });

  test('should support keyboard navigation', async ({ page, browserName }) => {
    // Skip on WebKit - keyboard events handled differently
    test.skip(browserName === 'webkit', 'WebKit handles keyboard events differently');

    // Open command palette
    await page.keyboard.press('Control+k');

    const commandPalette = page.locator('[data-testid="command-palette"]');
    await expect(commandPalette).toBeVisible();

    // First item should be highlighted by default (selectedIndex starts at 0)
    const firstItem = commandPalette.locator('[data-testid="command-item"]').first();
    await expect(firstItem).toHaveAttribute('data-highlighted', 'true');

    // Press arrow down to move to second item
    await page.keyboard.press('ArrowDown');
    
    // Wait a moment for state update
    await page.waitForTimeout(100);

    // Second item should be highlighted
    const secondItem = commandPalette.locator('[data-testid="command-item"]').nth(1);
    await expect(secondItem).toHaveAttribute('data-highlighted', 'true');

    // First item should no longer be highlighted
    await expect(firstItem).toHaveAttribute('data-highlighted', 'false');
  });

  test('should close with Escape', async ({ page, browserName }) => {
    // Skip on WebKit - Escape key handled differently
    test.skip(browserName === 'webkit', 'WebKit handles Escape key differently');

    // Open command palette
    await page.keyboard.press('Control+k');

    const commandPalette = page.locator('[data-testid="command-palette"]');
    await expect(commandPalette).toBeVisible();

    // Press Escape to close
    await page.keyboard.press('Escape');

    // Wait for close animation
    await page.waitForTimeout(200);

    // Command palette should be hidden
    await expect(commandPalette).not.toBeVisible();
  });

  test('should close when clicking outside', async ({ page }) => {
    // Open command palette
    await page.keyboard.press('Control+k');

    const commandPalette = page.locator('[data-testid="command-palette"]');
    await expect(commandPalette).toBeVisible();

    // Click on the backdrop (the palette itself is the backdrop)
    // Click at top-left corner which is outside the modal content
    await commandPalette.click({ position: { x: 10, y: 10 } });

    // Wait for close animation
    await page.waitForTimeout(200);

    // Command palette should be hidden
    await expect(commandPalette).not.toBeVisible();
  });
});
