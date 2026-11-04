"""MMOTU OTU_2d adapter: builds a manifest from the official release layout.

Layout (official code: configs/_base_/datasets/mmotu.py, mmseg/datasets/mmotu.py):
  OTU_2d/images/{id}.JPG
  OTU_2d/annotations/{id}_binary.PNG   binary lesion mask (0/1), used by official code
  OTU_2d/annotations/{id}.PNG          RGB semantic mask; colour encodes the category
  OTU_2d/train_cls.txt, val_cls.txt    "{id}.JPG  {class_index}"

Labels come only from *_cls.txt via label_policy. The official "val" list is the
paper's test split (Table I: training 1000 / testing 469).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ovascope_ml.data.hashing import sha256_file
from ovascope_ml.data.label_policy import (
    LABEL_POLICY_VERSION,
    MMOTU_REFERENCE_STANDARD,
    mmotu_index_to_class,
)
from ovascope_ml.data.manifest import DatasetSource, Manifest, ManifestRecord

# Palette from the official code's commented 8-class block, index-aligned with CLASSES.
MMOTU_PALETTE: tuple[tuple[int, int, int], ...] = (
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (64, 0, 64),
    (64, 64, 0), (64, 64, 64), (0, 128, 0), (0, 0, 128),
)

LICENSE_REFERENCE = (
    "MMOTU (Zhao et al., arXiv 2207.06799). No data licence published; code repository "
    "github.com/cv516Buaa/MMOTU_DS2Net is Apache-2.0. Local research use only; do not "
    "redistribute. See docs/DATASET_READINESS.md."
)


def mmotu_source(retrieved_at: str | None = None) -> DatasetSource:
    return DatasetSource(
        name="MMOTU OTU_2d",
        official_source=(
            "Google Drive folder linked from https://github.com/cv516Buaa/MMOTU_DS2Net "
            "(paper: arXiv 2207.06799, Pattern Recognition)"
        ),
        transport=(
            "Hugging Face mirror norayao/MMOTU @ d29379c60ac4072cfbf9c6f960505d43f98b3f0d, "
            "LFS SHA-256 verified; spot-checked against files from the official Drive folder"
        ),
        version="OTU_2d as released (1469 images)",
        retrieved_at=retrieved_at or datetime.now(UTC).date().isoformat(),
        license="Not stated for the images (repository code: Apache-2.0)",
        access_conditions="Publicly downloadable; no registration or data-use agreement observed",
        permitted_training_use="Not explicitly stated; used for local research development only",
        permitted_redistribution="Not established; images and masks are not redistributed",
        de_identified=(
            "Authors state images were manually cropped to remove patient private information. "
            "Burned-in device text, calipers, measurements and annotation labels remain."
        ),
        external_api_transmission_authorized=False,
        split_policy=(
            "Official MMOTU split: train_cls.txt -> development (1000), val_cls.txt -> test (469). "
            "Patient identifiers are not released, so patient-disjointness is unknown."
        ),
        split_unit="image",
        patient_independent=False,
    )


def read_cls_file(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 2:
            raise ValueError(f"{path.name}:{line_no}: expected '<file> <class>', got {line!r}")
        fname, idx = parts
        if not fname.upper().endswith(".JPG"):
            raise ValueError(f"{path.name}:{line_no}: unexpected image name {fname!r}")
        rows.append((fname[: -len(".JPG")], int(idx)))
    return rows


def build_manifest(root: Path, compute_hashes: bool = True) -> Manifest:
    otu = root / "OTU_2d"
    split_files = {"development": otu / "train_cls.txt", "test": otu / "val_cls.txt"}
    records: list[ManifestRecord] = []
    for split, cls_path in split_files.items():
        for image_stem, idx in read_cls_file(cls_path):
            cd = mmotu_index_to_class(idx)
            rel_img = f"OTU_2d/images/{image_stem}.JPG"
            rel_mask = f"OTU_2d/annotations/{image_stem}_binary.PNG"
            img_path = root / rel_img
            sha = sha256_file(img_path) if compute_hashes and img_path.exists() else "missing"
            records.append(
                ManifestRecord(
                    image_id=f"mmotu2d-{image_stem}",
                    patient_id=None,
                    study_id=None,
                    lesion_id=None,
                    image_path=rel_img,
                    image_sha256=sha,
                    source_dataset="MMOTU_OTU_2d",
                    source_center="Beijing Shijitan Hospital, Capital Medical University",
                    acquisition_date_if_permitted=None,
                    modality="ultrasound_2d_still",
                    view_type_if_known=None,
                    label_raw=f"{idx}:{cd.raw_code_name}",
                    label_normalized=cd.normalized,
                    reference_standard=MMOTU_REFERENCE_STANDARD,
                    histology_if_available=None,
                    stage_if_available=None,
                    mask_path_if_available=rel_mask if (root / rel_mask).exists() else None,
                    split=split,  # type: ignore[arg-type]
                    license_reference=LICENSE_REFERENCE,
                )
            )
    return Manifest(
        manifest_version="mmotu-otu2d-v1",
        label_policy_version=LABEL_POLICY_VERSION,
        source=mmotu_source(),
        records=records,
    )


def semantic_mask_path(root: Path, record: ManifestRecord) -> Path:
    stem = record.image_id.removeprefix("mmotu2d-")
    return root / "OTU_2d" / "annotations" / f"{stem}.PNG"
