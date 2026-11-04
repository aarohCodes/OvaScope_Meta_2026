"""API response schemas. File-system paths never appear in any response."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class StudyOut(BaseModel):
    id: str
    created_at: datetime
    retention_expires_at: datetime
    input_sha256: str
    image_format: str
    width: int
    height: int
    size_bytes: int
    external_transmission_consent: bool
    latest_job_id: str | None = None
    n_results: int = 0


class JobOut(BaseModel):
    id: str
    study_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    stage: str | None
    attempts: int
    max_attempts: int
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    reused: bool = Field(default=False, description="True when an idempotent request returned an existing job")


class ProviderOut(BaseModel):
    kind: str
    provider_id: str
    available: bool
    model_id: str | None = None
    model_version: str | None = None
    license: str | None = None
    meta_model: bool = False
    execution: str = "local"
    reason_unavailable: str | None = None
    action_required: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class CapabilitiesOut(BaseModel):
    notice: str
    mode: Literal["RESEARCH_INFERENCE", "NO_MODEL"]
    deployment: Literal["local", "public"]
    providers: list[ProviderOut]
    missing_requirements: list[dict[str, str]]
    worker: dict[str, Any]
    limits: dict[str, Any]
    accepted_formats: list[str]
    external_transmission: dict[str, Any]
    retention: dict[str, Any]
    budget: dict[str, Any]
    active_bundle: dict[str, Any] | None


class DeletionReceipt(BaseModel):
    study_id: str
    deleted: bool
    artifacts_removed: int
    results_removed: int
    jobs_removed: int
    annotations_removed: int


class AnnotationIn(BaseModel):
    polygon: list[list[float]] = Field(min_length=3, max_length=2000, description="[[x, y], ...] in source pixels")
    note: str | None = Field(default=None, max_length=500)


class AnnotationOut(BaseModel):
    id: str
    study_id: str
    polygon: list[list[float]]
    note: str | None
    created_at: datetime
    kind: str = "researcher_annotation"
    used_by_automatic_pipeline: bool = False
