import { expect, test } from "@playwright/test";
import { evaluationArtifact } from "./helpers";

test("evaluation page renders the actual evaluation artifact, or says none exists", async ({ page }) => {
  const ev = evaluationArtifact();
  await page.goto("/evaluation");
  if (!ev) {
    await expect(page.getByTestId("evaluation-unavailable")).toContainText("No metrics are shown");
    return;
  }
  const root = page.getByTestId("evaluation-page");
  await expect(root).toHaveAttribute("data-experiment", ev.experiment_id);
  await expect(page.getByTestId("eval-scope")).toContainText("External validation not performed");
  await expect(page.getByTestId("eval-scope")).toContainText("image");
  // the balanced accuracy shown for the primary pathway equals the artifact value
  const primary = ev.primary_pathway as string;
  const expected = (ev.models[primary].mmotu8.balanced_accuracy as number).toFixed(3);
  await expect(page.getByTestId("model-table")).toContainText(expected);
  await expect(page.getByTestId("ablations")).toContainText("What changed when this Meta component was removed?");
  await expect(page.getByTestId("segmentation-eval")).toContainText("Dice");
  await expect(page.getByTestId("failure-cases")).toBeVisible();
  // no early-stage claims without stage labels
  await expect(page.locator("body")).not.toContainText(/early[- ]stage (accuracy|sensitivity|performance)/i);
});
