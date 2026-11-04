import path from "node:path";
import fs from "node:fs";
import type { Page } from "@playwright/test";

export const REPO = path.resolve(__dirname, "../../..");
export const FIXTURE = path.join(REPO, "tests", "fixtures", "geometric", "ellipse_400x300.png");
export const MODE = process.env.E2E_MODE === "research" ? "research" : "no_model";

export async function uploadFixture(page: Page, file = FIXTURE) {
  await page.goto("/analyze");
  await page.getByTestId("file-input").setInputFiles(file);
  await page.waitForURL(/\/analyze\?study=[a-f0-9]{32}/);
}

export function evaluationArtifact(): Record<string, any> | null { // eslint-disable-line @typescript-eslint/no-explicit-any
  const p = path.join(REPO, "artifacts", "evaluation", "evaluation.json");
  return fs.existsSync(p) ? JSON.parse(fs.readFileSync(p, "utf8")) : null;
}
