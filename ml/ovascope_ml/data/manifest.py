"""Versioned dataset manifest.

Unknown values are explicit nulls. Patient, study and lesion identifiers are
null unless the source dataset releases them; they are never fabricated.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_SCHEMA_VERSION = "ovascope-manifest-1"

Split = Literal["development", "test"]


class ManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    image_id: str
    patient_id: str | None = None
    study_id: str | None = None
    lesion_id: str | None = None
    image_path: str = Field(description="POSIX path relative to the dataset root")
    image_sha256: str
    source_dataset: str
    source_center: str | None = None
    acquisition_date_if_permitted: str | None = None
    modality: str
    view_type_if_known: str | None = None
    label_raw: str
    label_normalized: str
    reference_standard: str
    histology_if_available: str | None = None
    stage_if_available: str | None = None
    mask_path_if_available: str | None = None
    split: Split | None = None
    license_reference: str
    width: int | None = None
    height: int | None = None


class DatasetSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    official_source: str
    transport: str
    version: str
    retrieved_at: str
    license: str
    access_conditions: str
    permitted_training_use: str
    permitted_redistribution: str
    de_identified: str
    external_api_transmission_authorized: bool
    split_policy: str
    split_unit: Literal["image", "lesion", "study", "patient"]
    patient_independent: bool


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = MANIFEST_SCHEMA_VERSION
    manifest_version: str
    label_policy_version: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source: DatasetSource
    records: list[ManifestRecord]

    def content_hash(self) -> str:
        """SHA-256 over records + source + label policy (excludes created_at)."""
        payload = {
            "schema_version": self.schema_version,
            "manifest_version": self.manifest_version,
            "label_policy_version": self.label_policy_version,
            "source": self.source.model_dump(exclude={"retrieved_at"}),
            "records": [r.model_dump() for r in sorted(self.records, key=lambda r: r.image_id)],
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()

    def split_hash(self) -> str:
        pairs = sorted((r.image_id, r.split or "") for r in self.records)
        return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()

    def by_split(self, split: Split) -> list[ManifestRecord]:
        return [r for r in self.records if r.split == split]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
