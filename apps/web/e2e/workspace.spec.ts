import { expect, test } from "@playwright/test";
import { MODE, uploadFixture } from "./helpers";

test.describe("persistent notice and honest readiness", () => {
  for (const route of ["/", "/analyze", "/evaluation", "/cards", "/settings"]) {
    test(`research notice is visible on ${route}`, async ({ page }) => {
      await page.goto(route);
      await expect(page.getByTestId("research-notice")).toHaveText("Research prototype. Not for diagnosis, screening, or treatment decisions.");
      const body = (await page.locator("body").innerText()).toLowerCase();
      for (const banned of ["cancer-free", "you have cancer", "safe to ignore", "probability of cancer"]) {
        expect(body).not.toContain(banned);
      }
    });
  }

  test("settings shows fail-closed external transmission", async ({ page }) => {
    await page.goto("/settings");
    await expect(page.getByTestId("external-policy")).toContainText("Disabled (fail closed)");
  });
});

test.describe("NO_MODEL mode", () => {
  test.skip(MODE !== "no_model", "runs with E2E_MODE=no_model");

  test("overview reports NO_MODEL and the missing classifier", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByTestId("overview-mode")).toHaveText("NO_MODEL");
    await expect(page.locator('[data-provider="classifier"]')).toHaveAttribute("data-available", "false");
  });

  test("upload -> real progress -> model_unavailable with no score -> delete", async ({ page }) => {
    await page.goto("/analyze");
    await expect(page.getByTestId("no-model-notice")).toBeVisible();
    await uploadFixture(page);
    await expect(page.getByTestId("result-status")).toHaveAttribute("data-status", "model_unavailable");
    await expect(page.getByTestId("missing-requirements")).toContainText("classifier");
    await expect(page.getByTestId("classification-panel")).toHaveAttribute("data-has-scores", "false");
    await expect(page.getByTestId("top-category")).toHaveCount(0);
    await expect(page.getByTestId("toggle-segmentation_mask")).toBeDisabled();
    await expect(page.getByTestId("viewer-canvas")).toHaveAttribute("data-view", /\d/);
    // viewer controls still work without a model
    const before = await page.getByTestId("viewer-zoom").innerText();
    await page.getByRole("button", { name: "Zoom in" }).click();
    await expect(page.getByTestId("viewer-zoom")).not.toHaveText(before);
    await page.getByTestId("viewer-reset").click();
    await expect(page.getByTestId("viewer-zoom")).toHaveText(before);
    // deletion
    const url = page.url();
    await page.getByTestId("delete-study").click();
    await page.getByTestId("confirm-delete").click();
    await page.waitForURL(/deleted=/);
    await page.goto(url);
    await expect(page.getByTestId("study-missing")).toBeVisible();
  });

  test("unsupported file is rejected with a clear error", async ({ page }) => {
    await page.goto("/analyze");
    await page.getByTestId("file-input").setInputFiles({ name: "scan.gif", mimeType: "image/png", buffer: Buffer.from("GIF89a" + "0".repeat(64)) });
    await expect(page.getByTestId("upload-error")).toContainText("UNSUPPORTED_FORMAT");
  });
});

test.describe("RESEARCH_INFERENCE mode (trained bundle)", () => {
  test.skip(MODE !== "research", "runs with E2E_MODE=research and an active trained bundle");

  test("real stages, separate panels, overlay controls", async ({ page }) => {
    await uploadFixture(page);
    // progress is reported by the worker in real time
    await expect(page.getByTestId("job-progress")).toHaveAttribute("data-status", /queued|running|succeeded/);
    await expect(page.getByTestId("result-status")).toHaveAttribute("data-status", /completed|completed_with_warnings|abstained/);
    await expect(page.getByTestId("segmentation-panel")).toBeVisible();
    await expect(page.getByTestId("classification-panel")).toHaveAttribute("data-has-scores", "true");
    await expect(page.getByTestId("score-heading")).toContainText("not calibrated probabilities");
    await expect(page.getByTestId("provenance-panel")).toContainText("Checkpoint SHA-256");
    const canvas = page.getByTestId("viewer-canvas");
    const segStatus = await page.getByTestId("segmentation-panel").getAttribute("data-seg-status");
    if (segStatus === "succeeded") {
      await expect(canvas).toHaveAttribute("data-overlays", /segmentation_mask:1:0\.85/);
      await page.getByTestId("toggle-segmentation_mask").uncheck();
      await expect(canvas).toHaveAttribute("data-overlays", /segmentation_mask:0:/);
      await page.getByTestId("toggle-segmentation_mask").check();
      await page.getByTestId("opacity-segmentation_mask").fill("0.3");
      await expect(canvas).toHaveAttribute("data-overlays", /segmentation_mask:1:0\.30/);
    } else {
      await expect(page.getByTestId("toggle-segmentation_mask")).toBeDisabled();
    }
    await page.getByTestId("toggle-attribution_map").check();
    await expect(canvas).toHaveAttribute("data-overlays", /attribution_map:1:/);
  });

  test("overlay registration survives a viewport resize", async ({ page }) => {
    await page.setViewportSize({ width: 1500, height: 950 });
    await uploadFixture(page);
    await expect(page.getByTestId("result-status")).toBeVisible();
    const canvas = page.getByTestId("viewer-canvas");
    const v1 = (await canvas.getAttribute("data-view"))!.split(",").map(Number);
    await page.setViewportSize({ width: 900, height: 800 });
    await expect.poll(async () => await canvas.getAttribute("data-view")).not.toBe(v1.join(","));
    const v2 = (await canvas.getAttribute("data-view"))!.split(",").map(Number);
    // Image and overlays share one transform; the image must remain fully on canvas after refit.
    expect(v2[0]).toBeGreaterThan(0);
    expect(v2[1]).toBeGreaterThanOrEqual(-1);
  });
});
