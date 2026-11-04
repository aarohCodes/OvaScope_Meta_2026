"""`ovascope-ml evaluate`: join test labels to frozen predictions and compute metrics.

Writes
  artifacts/experiments/<id>/metrics.json            full machine-readable metrics
  artifacts/experiments/<id>/predictions_test.csv    per-case predictions (local only)
  artifacts/experiments/<id>/evaluation_config.json  config text + artifact hashes
  artifacts/evaluation/evaluation.json               aggregate payload for the API/UI
  docs/generated/EVALUATION_REPORT.md                human-readable report
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ovascope_ml import paths
from ovascope_ml.data.label_policy import TARGETS, class_mapping_payload
from ovascope_ml.data.splits import split_statement
from ovascope_ml.evaluation import metrics as M
from ovascope_ml.evaluation.reference import test_labels, test_reference_masks
from ovascope_ml.models.geometry import Letterbox, dice_iou
from ovascope_ml.models.localizer import LocalizerParams, region_from_grid
from ovascope_ml.training import localizer_stage, sam_stage
from ovascope_ml.training.classifier_stage import VARIANTS
from ovascope_ml.training.common import CLASS_NAMES, HGS_INDEX, experiment_dir, load_manifest, log, stage_dir

EXTERNAL_VALIDATION = "External validation not performed."
SUBGROUP_MIN_N = 30


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hgs_auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    return M._safe_auc((y == HGS_INDEX).astype(int), s[:, HGS_INDEX])


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    exp = experiment_dir(cfg)
    selection = json.loads((exp / "selection.json").read_text(encoding="utf-8"))
    m = load_manifest(cfg)
    labels_by_id = test_labels(m)  # <- the only point where test labels enter
    with np.load(exp / "test_scores_unlabelled.npz", allow_pickle=False) as z:
        ids = [str(i) for i in z["image_ids"]]
        scores = {k: z[f"{k}__scores"] for k in VARIANTS}
        fallback = {k: z[f"{k}__fallback"] for k in VARIANTS}
    y = np.array([labels_by_id[i] for i in ids])
    yb = (y == HGS_INDEX).astype(int)
    nb, seed = cfg["bootstrap"]["n_resamples"], cfg["bootstrap"]["seed"]
    primary = selection["primary_pathway"]
    log(f"evaluate: {len(ids)} test images, primary pathway {primary}")

    models: dict[str, Any] = {}
    for name, (_, role) in VARIANTS.items():
        s = scores[name]
        sel = selection["variants"][name]
        mc = M.multiclass_metrics(y, s, CLASS_NAMES)
        mc["balanced_accuracy_ci"] = M.bootstrap_ci(M.balanced_accuracy, y, s, nb, seed)
        mc["macro_auroc_ci"] = M.bootstrap_ci(M.macro_auroc, y, s, nb, seed)
        hb = M.binary_metrics(yb, s[:, HGS_INDEX], sel["hgs_threshold"])
        hb["auroc_ci"] = M.bootstrap_ci(_hgs_auroc, y, s, nb, seed)
        models[name] = {
            "role": role,
            "dev_selection": {k: sel[k] for k in ("C", "temperature", "dev_oof_balanced_accuracy") if k in sel},
            "test_fallback_to_whole_image": int(fallback[name].sum()),
            "mmotu8": mc,
            "hgs_vs_other": hb,
            "abstention": M.abstention_metrics(y, s, sel["abstention_threshold"]),
            "risk_coverage": M.risk_coverage(y, s),
        }
    # Per-class OvR curves for the primary pathway (for plotting).
    curves = {}
    for k, name in enumerate(CLASS_NAMES):
        bm = M.binary_metrics((y == k).astype(int), scores[primary][:, k], 0.5)
        curves[name] = {
            "roc_curve": bm.get("roc_curve"),
            "pr_curve": bm.get("pr_curve"),
            "prevalence": bm["prevalence"],
            "auroc": bm["auroc"],
            "auprc": bm["auprc"],
        }
    models[primary]["per_class_curves"] = curves

    def ablation(question: str, a: str, b: str) -> dict[str, Any]:
        return {
            "question": question,
            "a": a,
            "b": b,
            "balanced_accuracy": M.paired_bootstrap_diff(M.balanced_accuracy, y, scores[a], scores[b], nb, seed),
            "macro_auroc": M.paired_bootstrap_diff(M.macro_auroc, y, scores[a], scores[b], nb, seed),
            "hgs_auroc": M.paired_bootstrap_diff(_hgs_auroc, y, scores[a], scores[b], nb, seed),
        }

    ablations = [
        ablation(
            "Does the DINOv2 representation beat a supervised ImageNet ResNet-50 (whole image)?",
            "dinov2_whole",
            "resnet50_imagenet",
        ),
        ablation(
            "Does the DINOv2 representation beat hand-crafted intensity/texture features?",
            "dinov2_whole",
            "handcrafted",
        ),
        ablation(
            "Does adding the automatic SAM region of interest help over the whole image?",
            "dinov2_roi_sam",
            "dinov2_whole",
        ),
        ablation(
            "Does SAM refinement help over the DINOv2 localiser box alone (ROI pathway)?",
            "dinov2_roi_sam",
            "dinov2_roi_coarse",
        ),
        ablation(
            "How far is the automatic ROI from an oracle reference-mask ROI (upper bound)?",
            "oracle_dinov2_roi_gt",
            "dinov2_roi_sam",
        ),
    ]

    segmentation = _segmentation(cfg, m, ids, y, nb, seed)
    sens = _sensitivity(ids, y, scores, primary)
    failures = _failure_cases(ids, y, scores[primary], selection["variants"][primary], fallback[primary])
    stmt = split_statement(m.records)

    payload: dict[str, Any] = {
        "schema_version": "ovascope-evaluation-1",
        "experiment_id": cfg["experiment_id"],
        "generated_at": datetime.now(UTC).isoformat(),
        "available": True,
        "scope": {
            "dataset": m.source.name,
            "evaluation_unit": "image",
            "split_unit": stmt.unit,
            "patient_independent": stmt.patient_independent,
            "statement": stmt.statement,
            "external_validation": EXTERNAL_VALIDATION,
            "test_split": "official MMOTU val list (469 images), fixed by the dataset authors",
            "n_test": len(ids),
            "test_class_distribution": {c: int((y == k).sum()) for k, c in enumerate(CLASS_NAMES)},
            "cohort_note": (
                "PPV/NPV and prevalence are specific to this enriched single-centre research "
                "cohort and do not transfer to screening or other populations."
            ),
            "ci_note": (
                "95% intervals: image-level percentile bootstrap. Patient identifiers are not "
                "available, so correlation between images of one patient is ignored and the "
                "intervals are likely too narrow."
            ),
        },
        "targets": {k: {"description": v.description, "classes": list(v.classes)} for k, v in TARGETS.items()},
        "class_mapping": class_mapping_payload(),
        "primary_pathway": primary,
        "primary_rule": selection["primary_rule"],
        "models": models,
        "ablations": ablations,
        "segmentation": segmentation,
        "sensitivity_analyses": sens,
        "failure_cases": failures,
        "early_stage": {"available": False, "reason": "MMOTU has no stage labels; stage is never inferred."},
        "novelty_heuristic": selection["novelty"],
    }
    prov = {
        "manifest_sha256": m.content_hash(),
        "split_sha256": m.split_hash(),
        "config_sha256": hashlib.sha256(cfg["_config_text"].encode()).hexdigest(),
        "selection_sha256": _sha(exp / "selection.json"),
        "test_scores_sha256": _sha(exp / "test_scores_unlabelled.npz"),
        "heads_sha256": {p.stem: _sha(p) for p in sorted((exp / "heads").glob("*.npz"))},
        "selection_frozen_at": selection["frozen_at"],
    }
    payload["provenance"] = prov
    (exp / "metrics.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    payload["metrics_sha256"] = _sha(exp / "metrics.json")
    (exp / "evaluation_config.json").write_text(
        json.dumps({"config_path": cfg["_config_path"], "config_text": cfg["_config_text"], **prov}, indent=1),
        encoding="utf-8",
    )
    _write_predictions(exp / "predictions_test.csv", ids, y, scores, fallback, primary, selection)
    out = paths.evaluation_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    gen = paths.REPO_ROOT / "docs" / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    from ovascope_ml.evaluation.report import render_evaluation_markdown

    (gen / "EVALUATION_REPORT.md").write_text(render_evaluation_markdown(payload), encoding="utf-8")
    log("evaluate: done")
    return payload


def _segmentation(cfg: dict[str, Any], m, ids: list[str], y: np.ndarray, nb: int, seed: int) -> dict[str, Any]:
    loc = localizer_stage.run(cfg)
    sam_dir = stage_dir(cfg, "sam")
    refs = test_reference_masks(m)
    params = LocalizerParams.load(stage_dir(cfg, "localizer") / "localizer_params.json")
    grids = np.load(stage_dir(cfg, "localizer") / "grid_test.npy")
    grid_row = {iid: k for k, iid in enumerate(loc["test_ids"])}
    rows = []
    for iid, label in zip(ids, y, strict=True):
        ref = refs.get(iid)
        if ref is None:
            continue
        h, w = ref.shape
        b = loc["boxes"][iid]
        coarse = np.zeros((h, w), bool)
        if b["status"] == "candidate_found":
            res = region_from_grid(grids[grid_row[iid]], Letterbox(w, h, cfg["backbone"]["img_size"]), params)
            coarse = res.coarse_mask if res.coarse_mask is not None else coarse
        auto_p = sam_dir / "auto" / f"{iid}.npz"
        sam_auto = sam_stage.load_mask(auto_p)[0] if auto_p.exists() else np.zeros((h, w), bool)
        orc_p = sam_dir / "eval_only" / "oracle_prompt" / f"{iid}.npz"
        sam_orc = sam_stage.load_mask(orc_p)[0] if orc_p.exists() else None
        rows.append(
            {
                "image_id": iid,
                "label": CLASS_NAMES[int(label)],
                "candidate": b["status"] == "candidate_found",
                "coarse": dice_iou(coarse, ref),
                "sam_auto": dice_iou(sam_auto, ref),
                "sam_oracle": dice_iou(sam_orc, ref) if sam_orc is not None else None,
            }
        )

    def summary(sel: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n": len(sel),
            "candidate_rate": float(np.mean([r["candidate"] for r in sel])) if sel else None,
        }
        for key in ("coarse", "sam_auto", "sam_oracle"):
            vals = [r[key] for r in sel if r[key] is not None]
            out[key] = {"dice": M.summarize_dice([v[0] for v in vals]), "iou": M.summarize_dice([v[1] for v in vals])}
        return out

    lesion_rows = [r for r in rows if r["label"] != "normal_ovary"]
    d_sam = np.array([r["sam_auto"][0] for r in rows])
    d_coarse = np.array([r["coarse"][0] for r in rows])
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(nb):
        idx = rng.integers(0, len(rows), len(rows))
        diffs.append(float(d_sam[idx].mean() - d_coarse[idx].mean()))
    return {
        "reference": "MMOTU binary masks (*_binary.PNG); for 'normal_ovary' images the mask outlines the ovary, not a lesion",
        "failure_counting": "images without an automatic candidate count as Dice 0",
        "localizer": "DINOv2 ViT-S/14 patch-token logistic localiser (development masks)",
        "sam_model": json.loads((sam_dir / "sam_summary.json").read_text(encoding="utf-8"))["model_id"],
        "all_test_images": summary(rows),
        "lesion_categories_only": summary(lesion_rows),
        "by_category": {c: summary([r for r in rows if r["label"] == c]) for c in CLASS_NAMES},
        "sam_minus_coarse_mean_dice": {
            "difference": float(d_sam.mean() - d_coarse.mean()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "method": "paired image-level bootstrap",
        },
        "oracle_prompt_note": "SAM prompted with the reference-mask box: an upper bound on prompt quality, not an automatic result.",
    }


def _sensitivity(ids: list[str], y: np.ndarray, scores: dict[str, np.ndarray], primary: str) -> dict[str, Any]:
    table = {
        r["image_id"]: r for r in json.loads((paths.processed_dir() / "image_table.json").read_text(encoding="utf-8"))
    }
    dev_clusters = {r["near_dup_cluster"] for r in table.values() if r["split"] == "development"}
    has_dup = np.array([table[i]["near_dup_cluster"] in dev_clusters for i in ids])
    colour = np.array([(table[i]["chroma_fraction"] or 0) > 0.01 for i in ids])
    s = scores[primary]

    def block(mask: np.ndarray, label: str) -> dict[str, Any]:
        n = int(mask.sum())
        per = {c: int((y[mask] == k).sum()) for k, c in enumerate(CLASS_NAMES)}
        if n < SUBGROUP_MIN_N:
            return {
                "subset": label,
                "n": n,
                "per_class_n": per,
                "suppressed": True,
                "reason": f"fewer than {SUBGROUP_MIN_N} images",
            }
        return {
            "subset": label,
            "n": n,
            "per_class_n": per,
            "suppressed": False,
            "balanced_accuracy": M.balanced_accuracy(y[mask], s[mask]),
            "accuracy": float((s[mask].argmax(1) == y[mask]).mean()),
            "note": "classes with few images make balanced accuracy unstable; see per_class_n",
        }

    return {
        "pathway": primary,
        "near_duplicate": {
            "description": "Test images whose perceptual-hash cluster also contains a development image",
            "with_dev_near_duplicate": block(has_dup, "with development near-duplicate"),
            "without_dev_near_duplicate": block(~has_dup, "without development near-duplicate"),
        },
        "colour_content": {
            "description": "Images with >1% colour pixels (colour Doppler and/or coloured annotations)",
            "colour_present": block(colour, "colour present"),
            "greyscale": block(~colour, "greyscale"),
        },
    }


def _failure_cases(ids: list[str], y: np.ndarray, s: np.ndarray, sel: dict[str, Any], fb: np.ndarray) -> dict[str, Any]:
    conf, pred = s.max(1), s.argmax(1)
    retained = conf >= sel["abstention_threshold"]
    wrong = (pred != y) & retained
    order = np.argsort(-conf)
    cw = [int(i) for i in order if wrong[i]][:6]
    ab = [int(i) for i in np.argsort(conf) if not retained[i]][:6]
    hgs_fn = [int(i) for i in range(len(y)) if y[i] == HGS_INDEX and s[i, HGS_INDEX] < sel["hgs_threshold"]]

    def item(i: int, why: str) -> dict[str, Any]:
        return {
            "image_id": ids[i],
            "reason": why,
            "true": CLASS_NAMES[int(y[i])],
            "highest_scoring": CLASS_NAMES[int(pred[i])],
            "top_score": float(conf[i]),
            "hgs_score": float(s[i, HGS_INDEX]),
            "used_whole_image_fallback": bool(fb[i]),
        }

    return {
        "selection_rule": (
            "Curated by fixed rule from the test set: highest-scoring wrong predictions "
            "that were NOT abstained, lowest-scoring abstained cases, and every 'High grade "
            "serous' image scored below the frozen operating threshold."
        ),
        "confident_errors": [item(i, "wrong and above abstention threshold") for i in cw],
        "abstained": [item(i, "below abstention threshold") for i in ab],
        "hgs_below_threshold": [item(i, "High grade serous image below frozen threshold") for i in hgs_fn],
    }


def _write_predictions(path: Path, ids, y, scores, fallback, primary, selection) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "image_id",
                "true_label",
                "variant",
                "predicted",
                "top_score",
                "abstained",
                "used_whole_image_fallback",
                *[f"score_{c}" for c in CLASS_NAMES],
            ]
        )
        for name, s in scores.items():
            thr = selection["variants"][name]["abstention_threshold"]
            for i, iid in enumerate(ids):
                w.writerow(
                    [
                        iid,
                        CLASS_NAMES[int(y[i])],
                        name,
                        CLASS_NAMES[int(s[i].argmax())],
                        f"{s[i].max():.6f}",
                        bool(s[i].max() < thr),
                        bool(fallback[name][i]),
                        *[f"{v:.6f}" for v in s[i]],
                    ]
                )
