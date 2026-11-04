"""`ovascope-ml export-bundle`: package an evaluated experiment for inference.

A bundle is a directory of NumPy/JSON files (no pickle) plus bundle.json, which
records the SHA-256 of each file. `checkpoint_sha256` is the SHA-256 of that
canonical file map, so any change to any file changes the checkpoint hash and
fails the integrity check at load time.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ovascope_ml import RESEARCH_NOTICE, __version__, paths
from ovascope_ml.data.label_policy import TARGETS, class_mapping_payload
from ovascope_ml.models.registry import get
from ovascope_ml.training.common import experiment_dir, load_manifest, stage_dir

BUNDLE_SCHEMA = "ovascope-bundle-1"


def file_map_hash(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def export(cfg: dict[str, Any]) -> Path:
    exp = experiment_dir(cfg)
    metrics_path = exp / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError("Run `ovascope-ml evaluate` first: bundles must reference an evaluation artifact.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selection = json.loads((exp / "selection.json").read_text(encoding="utf-8"))
    m = load_manifest(cfg)
    primary, fallback = selection["primary_pathway"], selection["fallback_pathway"]
    bundle_id = f"{cfg['experiment_id']}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    out = paths.bundles_dir() / bundle_id
    out.mkdir(parents=True, exist_ok=False)

    copies = {
        "localizer_head.npz": stage_dir(cfg, "localizer") / "localizer_head.npz",
        "localizer_params.json": stage_dir(cfg, "localizer") / "localizer_params.json",
        "head_dinov2_whole.npz": exp / "heads" / "dinov2_whole.npz",
        "head_dinov2_roi_sam.npz": exp / "heads" / "dinov2_roi_sam.npz",
        "novelty.npz": exp / "heads" / "novelty.npz",
    }
    files: dict[str, str] = {}
    for name, src in copies.items():
        shutil.copyfile(src, out / name)
        files[name] = hashlib.sha256((out / name).read_bytes()).hexdigest()

    bb, sam = get(cfg["backbone"]["key"]), get(cfg["sam"]["key"])
    pm = metrics["models"]
    bundle = {
        "schema_version": BUNDLE_SCHEMA,
        "bundle_id": bundle_id,
        "model_id": "ovascope-mmotu8-linear-heads",
        "model_version": f"{__version__}+{cfg['experiment_id']}",
        "created_at": datetime.now(UTC).isoformat(),
        "notice": RESEARCH_NOTICE,
        "classification_target": "mmotu8",
        "target_description": TARGETS["mmotu8"].description,
        "class_mapping": class_mapping_payload(),
        "pathways": {
            "dinov2_roi_sam": {
                "head": "head_dinov2_roi_sam.npz",
                "requires_segmentation": True,
                "selection": selection["variants"]["dinov2_roi_sam"],
            },
            "dinov2_whole": {
                "head": "head_dinov2_whole.npz",
                "requires_segmentation": False,
                "selection": selection["variants"]["dinov2_whole"],
            },
        },
        "primary_pathway": primary,
        "fallback_pathway": fallback,
        "pathway_rule": selection["primary_rule"]
        + " If the primary pathway needs a segmentation that fails, the independently evaluated "
        "whole-image pathway is used and the result says so.",
        "backbone": {
            "key": bb.key,
            "repo_id": bb.repo_id,
            "revision": bb.revision,
            "img_size": cfg["backbone"]["img_size"],
            "license": bb.license,
            "publisher": bb.publisher,
        },
        "roi": {"img_size": cfg["roi"]["img_size"], "margin": cfg["roi"]["margin"]},
        "segmentation": {
            "localizer_head": "localizer_head.npz",
            "localizer_params": "localizer_params.json",
            "sam_key": sam.key,
            "sam_repo_id": sam.repo_id,
            "sam_revision": sam.revision,
            "sam_license": sam.license,
            "multimask_output": cfg["sam"]["multimask_output"],
        },
        "calibration": {
            "status": "temperature_scaled_not_prevalence_calibrated",
            "description": (
                "Temperature fitted on development out-of-fold logits. Heads were trained with "
                "class-balanced weights, so scores are not calibrated probabilities of any "
                "category and must not be read as personal risk."
            ),
            "test_top_label_ece": {
                k: pm[k]["mmotu8"]["calibration"]["top_label_ece_10bin"] for k in ("dinov2_whole", "dinov2_roi_sam")
            },
        },
        "abstention": {
            "rule": "abstain when the highest category score is below the pathway threshold",
            "thresholds": {
                k: selection["variants"][k]["abstention_threshold"] for k in ("dinov2_whole", "dinov2_roi_sam")
            },
            "target_dev_coverage": cfg["abstention"]["target_dev_coverage"],
            "test_coverage": {k: pm[k]["abstention"]["coverage"] for k in ("dinov2_whole", "dinov2_roi_sam")},
        },
        "novelty": {"file": "novelty.npz", **selection["novelty"]},
        "evaluation_reference": {
            "experiment_id": cfg["experiment_id"],
            "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
            "evaluation_unit": metrics["scope"]["evaluation_unit"],
            "patient_independent": metrics["scope"]["patient_independent"],
            "external_validation": metrics["scope"]["external_validation"],
        },
        "training_data": {
            "dataset": m.source.name,
            "manifest_sha256": m.content_hash(),
            "split_sha256": m.split_hash(),
            "license": m.source.license,
            "redistribution": m.source.permitted_redistribution,
        },
        "files": files,
    }
    bundle["checkpoint_sha256"] = file_map_hash(files)
    (out / "bundle.json").write_text(json.dumps(bundle, indent=1), encoding="utf-8")
    (paths.bundles_dir() / "ACTIVE").write_text(bundle_id + "\n", encoding="utf-8")
    return out
