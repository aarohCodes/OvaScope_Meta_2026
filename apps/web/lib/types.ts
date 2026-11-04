// Types mirroring the OvaScope API (services/api/ovascope_api/schemas.py and
// ml/ovascope_ml/inference/result_schema.py). Keep in sync with schema_version.

export type Mode = "RESEARCH_INFERENCE" | "NO_MODEL";

export interface Provider {
  kind: string;
  provider_id: string;
  available: boolean;
  model_id: string | null;
  model_version: string | null;
  license: string | null;
  meta_model: boolean;
  execution: string;
  reason_unavailable: string | null;
  action_required: string | null;
  details: Record<string, unknown>;
}

export interface MissingRequirement {
  component: string;
  requirement: string;
  action: string;
}

export interface Capabilities {
  notice: string;
  mode: Mode;
  deployment: "local" | "public";
  providers: Provider[];
  missing_requirements: MissingRequirement[];
  worker: { online: boolean; last_seen_seconds_ago: number | null; inline: boolean | null; concurrency: number };
  limits: Record<string, number | string>;
  accepted_formats: string[];
  external_transmission: { allowed: boolean; policy: string; reason: string };
  retention: { hours: number; deletion: string };
  budget: { hosted_provider_monthly_cap_usd: number; hosted_calls_enabled: boolean };
  active_bundle: {
    bundle_id: string;
    model_id: string;
    model_version: string;
    checkpoint_sha256: string;
    primary_pathway: string;
    classification_target: string;
    calibration_status: string;
  } | null;
}

export interface Study {
  id: string;
  created_at: string;
  retention_expires_at: string;
  input_sha256: string;
  image_format: string;
  width: number;
  height: number;
  size_bytes: number;
  external_transmission_consent: boolean;
  latest_job_id: string | null;
  n_results: number;
}

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export interface Job {
  id: string;
  study_id: string;
  status: JobStatus;
  stage: string | null;
  attempts: number;
  max_attempts: number;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  reused: boolean;
}

export type ResultStatus =
  | "completed"
  | "completed_with_warnings"
  | "abstained"
  | "model_unavailable"
  | "unsupported_input"
  | "failed";

export type CalibrationStatus =
  | "not_applicable"
  | "uncalibrated"
  | "temperature_scaled_not_prevalence_calibrated";

export interface QualityCheck {
  id: string;
  category: "technical" | "image_quality" | "content" | "distribution_shift";
  status: "pass" | "warn" | "fail" | "info" | "not_run";
  message: string;
  value: number | null;
  validated: boolean;
}

export interface ArtifactRef {
  artifact_id: string;
  kind: "segmentation_mask" | "localizer_prior" | "attribution_map";
  media_type: string;
  width: number;
  height: number;
  source: string;
  description: string;
  limitations: string;
}

export interface SegmentationInfo {
  status: "succeeded" | "no_candidate_region" | "provider_unavailable" | "provider_failed" | "timeout" | "not_run";
  automatic: boolean;
  provider_id: string | null;
  model_id: string | null;
  localizer_model_id: string | null;
  prompt_source: string | null;
  prompt_box_xyxy: [number, number, number, number] | null;
  localizer_max_score: number | null;
  sam_predicted_iou: number | null;
  mask_area_fraction: number | null;
  message: string | null;
}

export interface CategoryScore {
  label: string;
  display: string;
  score: number;
}

export interface ClassificationInfo {
  pathway: "dinov2_roi_sam" | "dinov2_whole";
  pathway_reason: string;
  scores: CategoryScore[];
  highest_scoring_label: string;
  highest_scoring_display: string;
  score_semantics: string;
}

export interface AnalysisResult {
  schema_version: string;
  notice: string;
  study_id: string;
  status: ResultStatus;
  input_sha256: string;
  model_id: string | null;
  model_version: string | null;
  checkpoint_sha256: string | null;
  segmentation_provider: string | null;
  classification_target: string | null;
  class_mapping: Array<Record<string, unknown>> | null;
  model_score_or_null: Record<string, number> | null;
  calibration_status: CalibrationStatus;
  quality_checks: QualityCheck[];
  segmentation_artifact_or_null: ArtifactRef | null;
  localizer_artifact_or_null: ArtifactRef | null;
  attribution_artifact_or_null: ArtifactRef | null;
  abstention_status: "abstained" | "not_abstained" | "not_applicable";
  abstention_reason_or_null: string | null;
  warnings: string[];
  processing_times: Record<string, number>;
  evaluation_reference: Record<string, unknown> | null;
  created_at: string;
  segmentation: SegmentationInfo | null;
  classification: ClassificationInfo | null;
  missing_requirements: MissingRequirement[];
  error: { code: string; message: string; retryable: boolean } | null;
}

export interface ApiErrorBody {
  code: string;
  message: string;
  request_id?: string | null;
}

// Evaluation payload (ml/ovascope_ml/evaluation/evaluate.py). Loosely typed where nested.
export interface CI {
  point: number | null;
  ci95: [number, number] | null;
  n_resamples?: number;
  method?: string;
}

export interface PerClass {
  label: string;
  n: number;
  prevalence: number;
  sensitivity: number | null;
  specificity: number | null;
  ppv: number | null;
  npv: number | null;
  auroc_ovr: number | null;
  auprc_ovr: number | null;
}

export interface ModelEval {
  role: string;
  dev_selection: Record<string, number>;
  test_fallback_to_whole_image: number;
  mmotu8: {
    n: number;
    balanced_accuracy: number;
    accuracy: number;
    macro_auroc_ovr: number | null;
    per_class: PerClass[];
    confusion_matrix: { labels: string[]; rows_true_cols_pred: number[][] };
    calibration: {
      top_label_ece_10bin: number;
      reliability_bins: Array<{ lo: number; hi: number; n: number; mean_score: number; accuracy: number }>;
      multiclass_brier: number;
      nll: number;
    };
    balanced_accuracy_ci: CI;
    macro_auroc_ci: CI;
  };
  hgs_vs_other: {
    n: number;
    n_positive: number;
    prevalence: number;
    auroc: number | null;
    auprc: number | null;
    threshold: number;
    threshold_source: string;
    counts: { tp: number; fp: number; tn: number; fn: number };
    sensitivity: number | null;
    specificity: number | null;
    ppv: number | null;
    npv: number | null;
    roc_curve?: [number, number][];
    pr_curve?: [number, number][];
    auroc_ci: CI;
  };
  abstention: Record<string, number | string>;
  risk_coverage: Array<{ coverage: number; accuracy: number; balanced_accuracy: number }>;
  per_class_curves?: Record<string, { roc_curve: [number, number][] | null; pr_curve: [number, number][] | null; prevalence: number; auroc: number | null; auprc: number | null }>;
}

export interface Diff {
  difference: number | null;
  ci95: [number, number] | null;
  fraction_resamples_a_better: number | null;
  method: string;
}

export interface Evaluation {
  available: boolean;
  reason?: string;
  experiment_id?: string;
  generated_at?: string;
  scope?: Record<string, unknown> & {
    evaluation_unit: string;
    patient_independent: boolean;
    statement: string;
    external_validation: string;
    n_test: number;
    test_class_distribution: Record<string, number>;
    cohort_note: string;
    ci_note: string;
    test_split: string;
    dataset: string;
  };
  class_mapping?: Array<{ index: number; label: string; display: string }>;
  primary_pathway?: string;
  primary_rule?: string;
  models?: Record<string, ModelEval>;
  ablations?: Array<{ question: string; a: string; b: string; balanced_accuracy: Diff; macro_auroc: Diff; hgs_auroc: Diff }>;
  segmentation?: Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  sensitivity_analyses?: Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  failure_cases?: {
    selection_rule: string;
    confident_errors: FailureCase[];
    abstained: FailureCase[];
    hgs_below_threshold: FailureCase[];
  };
  early_stage?: { available: boolean; reason: string };
  provenance?: Record<string, unknown>;
  metrics_sha256?: string;
}

export interface FailureCase {
  image_id: string;
  reason: string;
  true: string;
  highest_scoring: string;
  top_score: number;
  hgs_score: number;
  used_whole_image_fallback: boolean;
}

export interface EvaluationResponse {
  notice: string;
  evaluation: Evaluation;
  data_audit: Record<string, any> | null; // eslint-disable-line @typescript-eslint/no-explicit-any
  mirror_verification: Record<string, unknown> | null;
}
