"""Manifest validation: files, decoding, duplicates, labels, groups and masks.

Produces (a) an aggregate report safe to commit (counts, no pixels, no paths)
and (b) a per-image table kept under data/processed (local only).
"""

from __future__ import annotations

import collections
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from ovascope_ml.data.hashing import (
    connected_components,
    dhash64,
    near_duplicate_pairs,
    phash64,
    sha256_file,
    sha256_pixels,
)
from ovascope_ml.data.imaging import ImageInputError, ImageLimits, load_mask_file, load_rgb_file
from ovascope_ml.data.manifest import Manifest, ManifestRecord

Severity = Literal["error", "warning", "info"]

# Dataset images are trusted more than uploads but still decoded defensively.
DATASET_LIMITS = ImageLimits(max_bytes=50 * 1024 * 1024, max_pixels=60_000_000, min_side=32)


@dataclass
class Issue:
    code: str
    severity: Severity
    detail: str
    image_ids: list[str] = field(default_factory=list)


@dataclass
class ImageRow:
    image_id: str
    split: str | None
    label: str
    width: int | None = None
    height: int | None = None
    pixel_sha256: str | None = None
    phash: int | None = None
    dhash: int | None = None
    chroma_fraction: float | None = None
    mask_area_fraction: float | None = None
    near_dup_cluster: int | None = None
    decode_ok: bool = False


@dataclass
class ValidationReport:
    n_records: int
    issues: list[Issue]
    rows: list[ImageRow]
    near_duplicate_pairs: list[tuple[str, str, int, int]]
    thresholds: dict[str, int]

    def counts(self) -> dict[str, int]:
        c: collections.Counter[str] = collections.Counter()
        for issue in self.issues:
            c[issue.code] += max(1, len(issue.image_ids))
        return dict(sorted(c.items()))

    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    def aggregate(self) -> dict[str, object]:
        """Commit-safe summary (no paths, no pixels; image IDs only for small issue sets)."""
        split_of = {r.image_id: r.split for r in self.rows}
        cross = [p for p in self.near_duplicate_pairs if split_of[p[0]] != split_of[p[1]]]
        clusters = collections.Counter(r.near_dup_cluster for r in self.rows if r.near_dup_cluster is not None)
        multi = [c for c, n in clusters.items() if n > 1]
        return {
            "n_records": self.n_records,
            "issue_counts": self.counts(),
            "issues": [
                {
                    "code": i.code,
                    "severity": i.severity,
                    "detail": i.detail,
                    "n_images": len(i.image_ids),
                    "example_image_ids": i.image_ids[:10],
                }
                for i in self.issues
            ],
            "near_duplicates": {
                "thresholds": self.thresholds,
                "n_pairs": len(self.near_duplicate_pairs),
                "n_cross_split_pairs": len(cross),
                "n_clusters_with_more_than_one_image": len(multi),
                "n_images_in_multi_image_clusters": sum(clusters[c] for c in multi),
            },
        }


def chroma_fraction(rgb: np.ndarray, threshold: int = 40) -> float:
    """Fraction of pixels with visible colour (max - min channel > threshold).

    In grey-scale B-mode this captures colour Doppler, coloured annotations and
    coloured UI markers. It is an audit statistic, not a quality score.
    """
    arr = rgb.astype(np.int16)
    spread = arr.max(axis=-1) - arr.min(axis=-1)
    return float((spread > threshold).mean())


def validate(
    manifest: Manifest,
    root: Path,
    *,
    max_phash: int = 6,
    max_dhash: int = 8,
    semantic_label_check: Callable[[ManifestRecord, np.ndarray | None], str | None] | None = None,
    semantic_mask_loader: Callable[[ManifestRecord], np.ndarray | None] | None = None,
) -> ValidationReport:
    issues: list[Issue] = []
    rows: list[ImageRow] = []

    def add(code: str, severity: Severity, detail: str, ids: list[str]) -> None:
        if ids:
            issues.append(Issue(code, severity, detail, ids))

    ids = [r.image_id for r in manifest.records]
    dup_ids = [k for k, n in collections.Counter(ids).items() if n > 1]
    add("duplicate_image_id", "error", "image_id appears more than once in the manifest", dup_ids)

    missing, invalid, sha_mismatch = [], [], []
    mask_missing, mask_dim_mismatch, mask_empty, mask_invalid = [], [], [], []
    label_conflict_semantic: list[str] = []
    for k, rec in enumerate(manifest.records):
        if k % 250 == 0:
            print(f"validate: {k}/{len(manifest.records)}", flush=True)
        row = ImageRow(rec.image_id, rec.split, rec.label_normalized)
        rows.append(row)
        path = root / rec.image_path
        if not path.exists():
            missing.append(rec.image_id)
            continue
        if rec.image_sha256 not in ("missing", sha256_file(path)):
            sha_mismatch.append(rec.image_id)
        try:
            decoded = load_rgb_file(path, DATASET_LIMITS)
        except ImageInputError as exc:
            invalid.append(f"{rec.image_id}:{exc.code.value}")
            continue
        row.decode_ok = True
        row.width, row.height = decoded.width, decoded.height
        row.pixel_sha256 = sha256_pixels(decoded.rgb)
        row.phash = phash64(decoded.rgb)
        row.dhash = dhash64(decoded.rgb)
        row.chroma_fraction = chroma_fraction(decoded.rgb)

        if rec.mask_path_if_available is None:
            mask_missing.append(rec.image_id)
        else:
            try:
                mask = load_mask_file(root / rec.mask_path_if_available)
            except Exception:
                mask_invalid.append(rec.image_id)
                mask = None
            if mask is not None:
                if mask.shape != (decoded.height, decoded.width):
                    mask_dim_mismatch.append(rec.image_id)
                elif not mask.any():
                    mask_empty.append(rec.image_id)
                else:
                    row.mask_area_fraction = float(mask.mean())
        if semantic_label_check is not None and semantic_mask_loader is not None:
            problem = semantic_label_check(rec, semantic_mask_loader(rec))
            if problem:
                label_conflict_semantic.append(f"{rec.image_id}:{problem}")

    add("missing_file", "error", "image file listed in manifest does not exist", missing)
    add("invalid_image", "error", "image failed defensive decoding (code appended)", invalid)
    add("sha256_mismatch", "error", "file hash differs from manifest", sha_mismatch)
    add("mask_missing", "warning", "no reference mask for image", mask_missing)
    add("mask_unreadable", "error", "reference mask could not be decoded", mask_invalid)
    add("mask_dimension_mismatch", "error", "mask size differs from image size", mask_dim_mismatch)
    add("mask_empty", "warning", "reference mask has no foreground", mask_empty)
    add(
        "label_conflict_semantic_mask",
        "error",
        "category encoded by the semantic-mask colour disagrees with the label file",
        label_conflict_semantic,
    )

    # Exact duplicates (pixel level, so re-encoded copies with identical pixels also match).
    by_pixels: dict[str, list[ImageRow]] = collections.defaultdict(list)
    for row in rows:
        if row.pixel_sha256:
            by_pixels[row.pixel_sha256].append(row)
    exact_groups = [g for g in by_pixels.values() if len(g) > 1]
    add(
        "exact_duplicate_pixels",
        "error",
        "identical decoded pixels in more than one record",
        [r.image_id for g in exact_groups for r in g],
    )
    add(
        "conflicting_labels",
        "error",
        "identical pixels carry different labels",
        [r.image_id for g in exact_groups if len({r.label for r in g}) > 1 for r in g],
    )
    add(
        "exact_duplicate_across_splits",
        "error",
        "identical pixels appear in both development and test",
        [r.image_id for g in exact_groups if len({r.split for r in g}) > 1 for r in g],
    )

    # Near duplicates.
    ok_rows = [r for r in rows if r.phash is not None and r.dhash is not None]
    raw_pairs = near_duplicate_pairs(
        [r.phash for r in ok_rows],  # type: ignore[misc]
        [r.dhash for r in ok_rows],  # type: ignore[misc]
        max_phash=max_phash,
        max_dhash=max_dhash,
    )
    clusters = connected_components(len(ok_rows), raw_pairs)
    for r, c in zip(ok_rows, clusters, strict=True):
        r.near_dup_cluster = c
    named_pairs = [(ok_rows[i].image_id, ok_rows[j].image_id, pd, dd) for i, j, pd, dd in raw_pairs]
    by_id = {r.image_id: r for r in ok_rows}
    cross_ids = sorted(
        {x for a, b, *_ in named_pairs for x in (a, b) if by_id[a].split != by_id[b].split}
    )
    add(
        "suspected_near_duplicate_across_splits",
        "warning",
        "perceptual-hash near duplicates span development and test (possible leakage; "
        "heuristic, not proof of same patient)",
        cross_ids,
    )
    label_mismatch_pairs = sorted(
        {x for a, b, *_ in named_pairs for x in (a, b) if by_id[a].label != by_id[b].label}
    )
    add(
        "near_duplicate_label_disagreement",
        "warning",
        "suspected near duplicates carry different labels (review: different lesions, "
        "annotation conflict, or hash false positive)",
        label_mismatch_pairs,
    )

    # Grouping identifiers.
    patient_ids = [r.patient_id for r in manifest.records]
    if all(p is None for p in patient_ids):
        add(
            "missing_grouping_identifiers",
            "warning",
            "no patient identifiers: patient-independent evaluation cannot be established; "
            "metrics are image-level and exploratory",
            ["<all>"],
        )
    else:
        per_patient = collections.Counter(p for p in patient_ids if p is not None)
        add(
            "multiple_images_per_patient",
            "info",
            "patients with more than one image (kept in one split)",
            [p for p, n in per_patient.items() if n > 1],
        )
        add(
            "missing_patient_id",
            "warning",
            "records without patient_id while others have one",
            [r.image_id for r in manifest.records if r.patient_id is None],
        )
        split_of: dict[str, set[str | None]] = collections.defaultdict(set)
        for mr in manifest.records:
            if mr.patient_id is not None:
                split_of[mr.patient_id].add(mr.split)
        add(
            "patient_group_leakage",
            "error",
            "patient appears in more than one split",
            [p for p, s in split_of.items() if len(s) > 1],
        )

    return ValidationReport(
        n_records=len(manifest.records),
        issues=issues,
        rows=rows,
        near_duplicate_pairs=named_pairs,
        thresholds={"max_phash_hamming": max_phash, "max_dhash_hamming": max_dhash},
    )


def rows_as_dicts(rows: list[ImageRow]) -> list[dict[str, object]]:
    out = []
    for r in rows:
        d = asdict(r)
        d["phash"] = f"{r.phash:016x}" if r.phash is not None else None
        d["dhash"] = f"{r.dhash:016x}" if r.dhash is not None else None
        out.append(d)
    return out
