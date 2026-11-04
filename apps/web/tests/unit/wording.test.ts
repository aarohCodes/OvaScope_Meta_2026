import { readFileSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  BANNED_PHRASES,
  calibrationExplanation,
  containsBannedPhrase,
  formatScore,
  resultStatusCopy,
  scoreHeading,
} from "@/lib/wording";
import type { ResultStatus } from "@/lib/types";

const STATUSES: ResultStatus[] = ["completed", "completed_with_warnings", "abstained", "model_unavailable", "unsupported_input", "failed"];

describe("wording guards", () => {
  it("never uses probability or risk wording for scores", () => {
    for (const cal of ["temperature_scaled_not_prevalence_calibrated", "uncalibrated", "not_applicable"] as const) {
      const h = scoreHeading(cal).toLowerCase();
      expect(h).not.toMatch(/\bprobabilit(y|ies) of\b|\brisk\b|%/);
    }
    expect(scoreHeading("uncalibrated")).toBe("Uncalibrated model scores");
    expect(calibrationExplanation("uncalibrated")).toContain("Uncalibrated model score");
  });

  it("formats scores without percent signs", () => {
    expect(formatScore(0.8734)).toBe("0.87");
    expect(formatScore(0.05)).not.toContain("%");
  });

  it("status copy contains no banned diagnostic phrases", () => {
    for (const s of STATUSES) {
      const c = resultStatusCopy(s);
      expect(containsBannedPhrase(`${c.title} ${c.body}`)).toBeNull();
    }
  });

  it("model-unavailable copy promises no score", () => {
    expect(resultStatusCopy("model_unavailable").body).toMatch(/no score/i);
  });

  it("no source file contains a banned phrase in user-facing strings", () => {
    const root = path.resolve(__dirname, "../..");
    const files: string[] = [];
    const walk = (d: string) => {
      for (const f of readdirSync(d)) {
        if (["node_modules", ".next", "tests", "playwright-report", "test-results", "e2e"].includes(f)) continue;
        const p = path.join(d, f);
        if (statSync(p).isDirectory()) walk(p);
        else if (/\.(tsx?|css)$/.test(f) && !p.endsWith("wording.ts")) files.push(p);
      }
    };
    walk(root);
    expect(files.length).toBeGreaterThan(5);
    for (const f of files) {
      const hit = containsBannedPhrase(readFileSync(f, "utf8"));
      expect(hit, `${path.relative(root, f)} contains "${hit}"`).toBeNull();
    }
  });

  it("allows negated disclaimers but blocks affirmative claims", () => {
    expect(containsBannedPhrase("Not clinically validated. No regulatory clearance.")).toBeNull();
    expect(containsBannedPhrase("This model is clinically validated")).toBe("clinically validated");
    expect(containsBannedPhrase("FDA approved workflow")).toBe("fda approved");
    expect(containsBannedPhrase("Result: cancer-free")).toBe("cancer-free");
    expect(BANNED_PHRASES).toContain("safe to ignore");
  });
});
