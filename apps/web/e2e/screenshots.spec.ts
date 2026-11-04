// Opt-in visual review: SCREENSHOTS=1 npx playwright test e2e/screenshots.spec.ts
// Images land in test-results/screens (gitignored). Uses only the synthetic geometric fixture.
import { expect, test } from "@playwright/test";
import { MODE, uploadFixture } from "./helpers";

test.skip(process.env.SCREENSHOTS !== "1", "set SCREENSHOTS=1 to capture review screenshots");

test("capture pages", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  for (const [route, name] of [["/", "overview"], ["/evaluation", "evaluation"], ["/cards", "cards"], ["/settings", "settings"]] as const) {
    await page.goto(route);
    await page.waitForLoadState("networkidle");
    await page.screenshot({ path: `test-results/screens/${MODE}-${name}.png`, fullPage: true });
  }
  await uploadFixture(page);
  await expect(page.getByTestId("result-status")).toBeVisible({ timeout: 120_000 });
  await page.screenshot({ path: `test-results/screens/${MODE}-workspace.png`, fullPage: true });
});
