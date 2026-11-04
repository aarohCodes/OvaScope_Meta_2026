// Display wording. Centralised so tests can guarantee the UI never uses
// diagnostic or probability language the evidence does not support.
import type { CalibrationStatus, ResultStatus, SegmentationInfo } from "./types";

export const RESEARCH_NOTICE = "Research prototype. Not for diagnosis, screening, or treatment decisions.";

/** Diagnostic phrases that must never appear in rendered UI copy, in any form. */
export const BANNED_PHRASES = [
  "cancer-free",
  "cancer free",
  "you have cancer",
  "safe to ignore",
  "benign, no follow-up",
  "risk of cancer",
  "probability of cancer",
  "chance of cancer",
  "diagnosis:",
];

/** Unsupported claims: allowed only when explicitly negated ("not clinically validated"). */
export const CLAIM_PHRASES = ["clinically validated", "fda approved", "fda cleared", "outperforms doctors", "detects stage i"];

const NEGATION = /\b(not|no|never|without)\s+(\w+\s+)?$/;

export function containsBannedPhrase(text: string): string | null {
  const t = text.toLowerCase();
  const hard = BANNED_PHRASES.find((p) => t.includes(p));
  if (hard) return hard;
  for (const p of CLAIM_PHRASES) {
    let i = t.indexOf(p);
    while (i >= 0) {
      if (!NEGATION.test(t.slice(Math.max(0, i - 24), i))) return p;
      i = t.indexOf(p, i + 1);
    }
  }
  return null;
}

/** Heading for the score panel. Never "probability" or "risk". */
export function scoreHeading(cal: CalibrationStatus): string {
  switch (cal) {
    case "temperature_scaled_not_prevalence_calibrated":
      return "Model scores (temperature-scaled; not calibrated probabilities)";
    case "uncalibrated":
      return "Uncalibrated model scores";
    default:
      return "Model scores";
  }
}

export function calibrationExplanation(cal: CalibrationStatus): string {
  switch (cal) {
    case "temperature_scaled_not_prevalence_calibrated":
      return "Scores were temperature-scaled on development data. The classifier was trained with class-balanced weights, so scores are not probabilities of any category and are not a personal risk estimate.";
    case "uncalibrated":
      return "Uncalibrated model score. Not a probability and not a personal risk estimate.";
    default:
      return "No classifier score was produced.";
  }
}

export function formatScore(score: number): string {
  // Two decimals, never a percent sign: a percentage reads as a probability.
  return score.toFixed(2);
}

export interface StatusCopy {
  title: string;
  body: string;
  tone: "neutral" | "accent" | "warning" | "error";
}

export function resultStatusCopy(status: ResultStatus): StatusCopy {
  switch (status) {
    case "completed":
      return { title: "Analysis completed", body: "Classifier output and segmentation are shown separately below.", tone: "accent" };
    case "completed_with_warnings":
      return { title: "Completed with warnings", body: "Review the warnings before interpreting any output.", tone: "warning" };
    case "abstained":
      return {
        title: "Model abstained",
        body: "The highest category score fell below the abstention threshold frozen on development data. No category is reported.",
        tone: "warning",
      };
    case "model_unavailable":
      return {
        title: "Model unavailable",
        body: "No trained classifier is configured, so no score or category is produced. The missing requirements are listed below.",
        tone: "neutral",
      };
    case "unsupported_input":
      return { title: "Unsupported input", body: "The file could not be processed. Only PNG and JPEG still images are supported.", tone: "error" };
    case "failed":
      return { title: "Analysis failed", body: "A model or provider failed. No result is shown in its place.", tone: "error" };
  }
}

export function segmentationCopy(seg: SegmentationInfo | null): string {
  if (!seg) return "Segmentation was not run.";
  switch (seg.status) {
    case "succeeded":
      return "Automatic SAM 2.1 mask from a DINOv2-localiser box prompt.";
    case "no_candidate_region":
      return "The localiser found no candidate region, so SAM was not prompted and no mask is shown.";
    case "provider_unavailable":
      return "The SAM provider is unavailable; no mask was produced.";
    case "provider_failed":
      return "SAM failed on this image; no mask is shown.";
    case "timeout":
      return "SAM exceeded its time limit; no mask is shown.";
    case "not_run":
      return "Segmentation was not run.";
  }
}

export const STAGES = ["queued", "validating", "representation", "segmentation", "classification", "attribution", "done"] as const;

export function stageLabel(stage: string | null, status: string): string {
  if (status === "queued") return "Queued";
  if (status === "succeeded") return "Done";
  if (status === "failed") return "Failed";
  if (status === "cancelled") return "Cancelled";
  const map: Record<string, string> = {
    claimed: "Starting",
    validating: "Validating input",
    representation: "DINOv2 representation",
    segmentation: "Localisation + SAM 2.1",
    classification: "Classifier",
    attribution: "Attribution map",
  };
  return stage ? (map[stage] ?? stage) : "Running";
}

export function categoryDisplay(label: string): string {
  const map: Record<string, string> = {
    chocolate_cyst: "Chocolate cyst (endometrioma)",
    serous_cystadenoma: "Serous cystadenoma",
    teratoma: "Teratoma",
    theca_cell_tumor: "Theca cell tumour",
    simple_cyst: "Simple cyst",
    normal_ovary: "Normal ovary",
    mucinous_cystadenoma: "Mucinous cystadenoma",
    high_grade_serous: "High-grade serous (MMOTU category)",
  };
  return map[label] ?? label;
}

export const MODEL_NAMES: Record<string, string> = {
  majority: "Majority class",
  handcrafted: "Hand-crafted features",
  resnet50_imagenet: "ImageNet ResNet-50",
  dinov2_whole: "DINOv2 · whole image",
  dinov2_roi_coarse: "DINOv2 + localiser box (no SAM)",
  dinov2_roi_sam: "DINOv2 + SAM 2.1 ROI",
  oracle_dinov2_roi_gt: "Oracle: reference-mask ROI",
};
