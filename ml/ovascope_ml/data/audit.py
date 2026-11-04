"""`ovascope-ml data-audit`: manifest + validation + shortcut audit.

Outputs
  data/processed/manifest_mmotu_otu2d_v1.json   (local only; gitignored)
  data/processed/image_table.json               (local only; per-image audit stats)
  artifacts/evaluation/data_audit.json          (aggregate; served by the API)
  docs/generated/DATA_AUDIT.md                  (aggregate; human-readable)
"""

from __future__ import annotations

import collections
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ovascope_ml import paths
from ovascope_ml.data.label_policy import MMOTU_CLASSES, TARGETS
from ovascope_ml.data.manifest import Manifest, ManifestRecord
from ovascope_ml.data.mmotu import MMOTU_PALETTE, build_manifest, semantic_mask_path
from ovascope_ml.data.shortcut_audit import metadata_probe, per_class_artifact_table
from ovascope_ml.data.splits import grouped_cv_folds, group_keys, split_statement
from ovascope_ml.data.validation import ValidationReport, rows_as_dicts, validate

MANIFEST_NAME = "manifest_mmotu_otu2d_v1.json"


def _semantic_loader(root: Path):
    from PIL import Image

    def load(rec: ManifestRecord) -> np.ndarray | None:
        p = semantic_mask_path(root, rec)
        if not p.exists():
            return None
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB"))

    return load


def mmotu_semantic_check(rec: ManifestRecord, sem: np.ndarray | None) -> str | None:
    if sem is None:
        return "semantic_mask_missing"
    packed = np.unique(sem.reshape(-1, 3).astype(np.int32) @ np.array([65536, 256, 1], dtype=np.int32))
    colours = {(int(v) >> 16 & 255, int(v) >> 8 & 255, int(v) & 255) for v in packed} - {(0, 0, 0)}
    if not colours:
        return "semantic_mask_empty"
    idxs = set()
    for c in colours:
        if c not in MMOTU_PALETTE:
            return f"unknown_colour_{c}"
        idxs.add(MMOTU_PALETTE.index(c))
    label_idx = int(rec.label_raw.split(":", 1)[0])
    if idxs != {label_idx}:
        return f"palette_{sorted(idxs)}_vs_label_{label_idx}"
    return None


def run_audit(root: Path | None = None, out_processed: Path | None = None) -> dict[str, Any]:
    root = root or paths.mmotu_root()
    out_processed = out_processed or paths.processed_dir()
    manifest = build_manifest(root)
    report = validate(
        manifest,
        root,
        semantic_label_check=mmotu_semantic_check,
        semantic_mask_loader=_semantic_loader(root),
    )
    out_processed.mkdir(parents=True, exist_ok=True)
    manifest.save(out_processed / MANIFEST_NAME)
    (out_processed / "image_table.json").write_text(
        json.dumps(rows_as_dicts(report.rows), indent=0), encoding="utf-8"
    )
    summary = summarize(manifest, report)
    eval_dir = paths.evaluation_dir()
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "data_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    gen = paths.REPO_ROOT / "docs" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "DATA_AUDIT.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def summarize(manifest: Manifest, report: ValidationReport) -> dict[str, Any]:
    class_names = [c.normalized for c in MMOTU_CLASSES]
    rows_by_id = {r.image_id: r for r in report.rows}
    dist: dict[str, dict[str, int]] = {}
    for split in ("development", "test"):
        c = collections.Counter(r.label_normalized for r in manifest.by_split(split))
        dist[split] = {name: c.get(name, 0) for name in class_names}

    dev = manifest.by_split("development")
    test = manifest.by_split("test")
    cluster = {r.image_id: r.near_dup_cluster for r in report.rows if r.near_dup_cluster is not None}
    groups = group_keys(dev, cluster)
    dev_labels = [class_names.index(r.label_normalized) for r in dev]
    folds = grouped_cv_folds(dev_labels, groups)
    dev_rows = [rows_by_id[r.image_id] for r in dev]
    test_rows = [rows_by_id[r.image_id] for r in test]
    probe = metadata_probe(
        dev_rows, dev_labels, folds, test_rows, [class_names.index(r.label_normalized) for r in test]
    )
    widths = [r.width for r in report.rows if r.width]
    heights = [r.height for r in report.rows if r.height]
    stmt = split_statement(manifest.records)
    has_stage = any(r.stage_if_available for r in manifest.records)
    has_histology = any(r.histology_if_available for r in manifest.records)
    return {
        "schema_version": "ovascope-data-audit-1",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": manifest.source.model_dump(),
        "manifest_version": manifest.manifest_version,
        "manifest_sha256": manifest.content_hash(),
        "split_sha256": manifest.split_hash(),
        "label_policy_version": manifest.label_policy_version,
        "targets": {k: {"description": v.description, "classes": list(v.classes)} for k, v in TARGETS.items()},
        "class_distribution": dist,
        "n_images": len(manifest.records),
        "n_patients": None,
        "n_patients_note": (
            "Not released. The paper reports 294 patients for the full MMOTU dataset "
            "(OTU_2d + CEUS, 1,639 images); the OTU_2d patient count is not stated."
        ),
        "split": {
            "unit": stmt.unit,
            "patient_independent": stmt.patient_independent,
            "cv_grouping": stmt.grouping_used_for_cv,
            "statement": stmt.statement,
            "n_cv_folds": int(folds.max() + 1),
        },
        "stage_labels_available": has_stage,
        "histology_labels_available": has_histology,
        "early_stage_analysis": (
            "blocked: no stage labels in MMOTU" if not has_stage else "possible; see evaluation"
        ),
        "borderline_category_present": False,
        "image_dimensions": {
            "width_min": min(widths) if widths else None,
            "width_max": max(widths) if widths else None,
            "height_min": min(heights) if heights else None,
            "height_max": max(heights) if heights else None,
        },
        "validation": report.aggregate(),
        "shortcut_audit": {
            "metadata_probe": probe,
            "per_class": per_class_artifact_table(report.rows, class_names),
            "burned_in_content": (
                "The dataset paper states most images carry expert-drawn symbols (lines, hands, "
                "characters, arrows); sample figures show device settings text, calipers, "
                "measurements and labels such as 'CY' and 'LOV'. These were not removed."
            ),
        },
    }


def render_markdown(s: dict[str, Any]) -> str:
    v = s["validation"]
    probe = s["shortcut_audit"]["metadata_probe"]
    lines = [
        "# MMOTU OTU_2d data audit (generated)",
        "",
        "Research prototype. Not for diagnosis, screening, or treatment decisions.",
        "",
        f"Generated {s['generated_at']} by `ovascope-ml data-audit`. Aggregate values only.",
        "",
        f"- Manifest `{s['manifest_version']}` sha256 `{s['manifest_sha256']}`",
        f"- Split sha256 `{s['split_sha256']}`; label policy `{s['label_policy_version']}`",
        f"- Images: {s['n_images']}; patients: not released ({s['n_patients_note']})",
        f"- Split unit: **{s['split']['unit']}**; patient-independent: **{s['split']['patient_independent']}**",
        f"- {s['split']['statement']}",
        f"- Stage labels: {s['stage_labels_available']}; early-stage analysis: {s['early_stage_analysis']}",
        "",
        "## Class distribution",
        "",
        "| Category | Development | Test |",
        "|---|---:|---:|",
    ]
    dist = s["class_distribution"]
    for name in dist["development"]:
        lines.append(f"| {name} | {dist['development'][name]} | {dist['test'][name]} |")
    lines += ["", "## Validation issues", "", "| Code | Severity | Images | Detail |", "|---|---|---:|---|"]
    for i in v["issues"]:
        lines.append(f"| {i['code']} | {i['severity']} | {i['n_images']} | {i['detail']} |")
    nd = v["near_duplicates"]
    lines += [
        "",
        "## Near duplicates (perceptual hash; heuristic)",
        "",
        f"- Thresholds: {nd['thresholds']}",
        f"- Suspected pairs: {nd['n_pairs']} (cross-split: {nd['n_cross_split_pairs']})",
        f"- Clusters with >1 image: {nd['n_clusters_with_more_than_one_image']} "
        f"covering {nd['n_images_in_multi_image_clusters']} images",
        "",
        "## Shortcut audit: metadata-only probe",
        "",
        f"Features: {', '.join(probe['features'])}. Model: {probe['model']}.",
        "",
        "| Target | Dev grouped-CV balanced acc. | Test balanced acc. | Chance |",
        "|---|---:|---:|---:|",
        f"| mmotu8 | {probe['mmotu8']['dev_grouped_cv_balanced_accuracy']:.3f} | "
        f"{probe['mmotu8']['test_balanced_accuracy']:.3f} | {probe['chance_balanced_accuracy_mmotu8']:.3f} |",
        f"| hgs_vs_other | {probe['hgs_vs_other']['dev_grouped_cv_balanced_accuracy']:.3f} | "
        f"{probe['hgs_vs_other']['test_balanced_accuracy']:.3f} | 0.500 |",
        "",
        probe["interpretation"],
        "",
        "| Category | n | Colour pixels >1% | Median colour fraction | Top frame-size clusters |",
        "|---|---:|---:|---:|---|",
    ]
    for row in s["shortcut_audit"]["per_class"]:
        lines.append(
            f"| {row['label']} | {row['n']} | {row['fraction_with_colour_pixels_gt_1pct']:.2f} | "
            f"{row['median_colour_pixel_fraction']:.4f} | "
            + ", ".join(f"{k} ({n})" for k, n in row["top_frame_size_clusters_20px"])
            + " |"
        )
    lines += ["", s["shortcut_audit"]["burned_in_content"], ""]
    return "\n".join(lines)
