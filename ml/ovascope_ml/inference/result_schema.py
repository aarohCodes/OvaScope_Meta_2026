"""Versioned analysis-result contract shared by the engine, worker and API."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ovascope_ml import RESEARCH_NOTICE

RESULT_SCHEMA_VERSION = "ovascope-result-1"


class ResultStatus(str, Enum):
    completed = "completed"
    completed_with_warnings = "completed_with_warnings"
    abstained = "abstained"
    model_unavailable = "model_unavailable"
    unsupported_input = "unsupported_input"
    failed = "failed"


class CalibrationStatus(str, Enum):
    not_applicable = "not_applicable"
    uncalibrated = "uncalibrated"
    temperature_scaled_not_prevalence_calibrated = "temperature_scaled_not_prevalence_calibrated"


class QualityCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: Literal["technical", "image_quality", "content", "distribution_shift"]
    status: Literal["pass", "warn", "fail", "info", "not_run"]
    message: str
    value: float | None = None
    validated: bool = Field(
        default=False, description="True only for checks with evidence of clinical validity (none here)"
    )


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: str
    kind: Literal["segmentation_mask", "localizer_prior", "attribution_map"]
    media_type: str = "image/png"
    width: int
    height: int
    source: str
    description: str
    limitations: str


class SegmentationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["succeeded", "no_candidate_region", "provider_unavailable", "provider_failed", "timeout", "not_run"]
    automatic: bool = True
    provider_id: str | None = None
    model_id: str | None = None
    localizer_model_id: str | None = None
    prompt_source: str | None = None
    prompt_box_xyxy: list[int] | None = None
    localizer_max_score: float | None = None
    sam_predicted_iou: float | None = Field(default=None, description="SAM's own IoU estimate; not validated")
    mask_area_fraction: float | None = None
    message: str | None = None


class CategoryScore(BaseModel):
    label: str
    display: str
    score: float


class ClassificationInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pathway: Literal["dinov2_roi_sam", "dinov2_whole"]
    pathway_reason: str
    scores: list[CategoryScore]
    highest_scoring_label: str
    highest_scoring_display: str
    score_semantics: str


class MissingRequirement(BaseModel):
    component: str
    requirement: str
    action: str


class ErrorInfo(BaseModel):
    code: str
    message: str
    retryable: bool = False


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = RESULT_SCHEMA_VERSION
    notice: str = RESEARCH_NOTICE
    study_id: str
    status: ResultStatus
    input_sha256: str
    model_id: str | None = None
    model_version: str | None = None
    checkpoint_sha256: str | None = None
    segmentation_provider: str | None = None
    classification_target: str | None = None
    class_mapping: list[dict[str, object]] | None = None
    model_score_or_null: dict[str, float] | None = None
    calibration_status: CalibrationStatus = CalibrationStatus.not_applicable
    quality_checks: list[QualityCheck] = Field(default_factory=list)
    segmentation_artifact_or_null: ArtifactRef | None = None
    localizer_artifact_or_null: ArtifactRef | None = None
    attribution_artifact_or_null: ArtifactRef | None = None
    abstention_status: Literal["abstained", "not_abstained", "not_applicable"] = "not_applicable"
    abstention_reason_or_null: str | None = None
    warnings: list[str] = Field(default_factory=list)
    processing_times: dict[str, float] = Field(default_factory=dict, description="milliseconds per stage")
    evaluation_reference: dict[str, object] | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    segmentation: SegmentationInfo | None = None
    classification: ClassificationInfo | None = None
    missing_requirements: list[MissingRequirement] = Field(default_factory=list)
    error: ErrorInfo | None = None
