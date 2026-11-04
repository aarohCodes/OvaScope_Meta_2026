import { defineConfig, devices } from "@playwright/test";
import path from "node:path";

// E2E runs against the REAL API and the production web build.
//   E2E_MODE=no_model (default): forces NO_MODEL; verifies honest unavailable states.
//   E2E_MODE=research: uses the ACTIVE trained bundle (requires `make train evaluate bundle`).
const mode = process.env.E2E_MODE === "research" ? "research" : "no_model";
const repo = path.resolve(__dirname, "../..");
const py =
  process.env.OVASCOPE_PYTHON ??
  (process.platform === "win32" ? path.join(repo, ".venv", "Scripts", "python.exe") : path.join(repo, ".venv", "bin", "python"));
const scratch = path.join(__dirname, "test-results", `e2e-data-${mode}`);
const API_PORT = 8765;
const WEB_PORT = 3100;

export default defineConfig({
  testDir: "./e2e",
  timeout: mode === "research" ? 180_000 : 60_000,
  expect: { timeout: mode === "research" ? 120_000 : 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: `http://127.0.0.1:${WEB_PORT}`, trace: "retain-on-failure", ...devices["Desktop Chrome"] },
  metadata: { mode },
  webServer: [
    {
      command: `"${py}" -m uvicorn ovascope_api.main:get_app --factory --host 127.0.0.1 --port ${API_PORT}`,
      cwd: repo,
      url: `http://127.0.0.1:${API_PORT}/health`,
      reuseExistingServer: false,
      timeout: 120_000,
      env: {
        OVASCOPE_DATABASE_URL: `sqlite:///${path.join(scratch, "e2e.db").replaceAll("\\", "/")}`,
        OVASCOPE_ARTIFACT_DIR: path.join(scratch, "store"),
        OVASCOPE_INFERENCE_MODE: mode === "research" ? "auto" : "no_model",
        OVASCOPE_INLINE_WORKER: "true",
        OVASCOPE_SERVE_EVAL_IMAGES: "false",
        HF_HUB_OFFLINE: "1",
      },
    },
    {
      command: `npx next start --port ${WEB_PORT}`,
      cwd: __dirname,
      url: `http://127.0.0.1:${WEB_PORT}`,
      reuseExistingServer: false,
      timeout: 120_000,
      env: { OVASCOPE_API_INTERNAL_URL: `http://127.0.0.1:${API_PORT}`, NEXT_TELEMETRY_DISABLED: "1" },
    },
  ],
});
