"""Human-readable evaluation report rendered from the machine-readable payload."""

from __future__ import annotations

from typing import Any


def _f(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _ci(d: dict[str, Any] | None) -> str:
    if not d or d.get("point") is None:
        return "n/a"
    ci = d.get("ci95")
    return f"{d['point']:.3f}" + (f" ({ci[0]:.3f}-{ci[1]:.3f})" if ci else "")


NAMES = {
    "majority": "Majority class",
    "handcrafted": "Hand-crafted intensity/texture + LR",
    "resnet50_imagenet": "ImageNet ResNet-50 (frozen) + LR",
    "dinov2_whole": "DINOv2 ViT-S/14 whole image + LR",
    "dinov2_roi_coarse": "DINOv2 whole + localiser-box ROI (no SAM)",
    "dinov2_roi_sam": "DINOv2 whole + automatic SAM 2.1 ROI",
    "oracle_dinov2_roi_gt": "ORACLE: DINOv2 whole + reference-mask ROI",
}


def render_evaluation_markdown(p: dict[str, Any]) -> str:
    sc = p["scope"]
    L = [
        "# OvaScope evaluation report (generated)",
        "",
        "Research prototype. Not for diagnosis, screening, or treatment decisions.",
        "",
        f"Experiment `{p['experiment_id']}`, generated {p['generated_at']}.",
        "",
        "## Scope",
        "",
        f"- Dataset: {sc['dataset']}; test split: {sc['test_split']}; n = {sc['n_test']} images.",
        f"- Evaluation unit: **{sc['evaluation_unit']}**. Patient-independent: **{sc['patient_independent']}**.",
        f"- {sc['statement']}",
        f"- **{sc['external_validation']}**",
        f"- {sc['cohort_note']}",
        f"- {sc['ci_note']}",
        f"- Early-stage analysis: {p['early_stage']['reason']}",
        "",
        f"Primary deployed pathway: `{p['primary_pathway']}`. Rule: {p['primary_rule']}",
        "",
        "## Primary target `mmotu8` (8 MMOTU categories)",
        "",
        "| Model | Role | Balanced acc. (95% CI) | Macro AUROC (95% CI) | Accuracy | Top-label ECE | Abstention coverage | Retained acc. |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for name, mdl in p["models"].items():
        mc, ab = mdl["mmotu8"], mdl["abstention"]
        L.append(
            f"| {NAMES.get(name, name)} | {mdl['role']} | {_ci(mc['balanced_accuracy_ci'])} | {_ci(mc['macro_auroc_ci'])} | "
            f"{_f(mc['accuracy'])} | {_f(mc['calibration']['top_label_ece_10bin'])} | {_f(ab['coverage'])} | {_f(ab.get('accuracy_retained'))} |"
        )
    L += [
        "",
        "The oracle row uses reference masks at test time and is an upper bound, not an automatic result.",
        "",
        "## Secondary target `hgs_vs_other`",
        "",
        "Score for the MMOTU 'High grade serous' category vs all other categories, threshold frozen on development data.",
        "",
        "| Model | n pos / n | AUROC (95% CI) | AUPRC (prevalence) | Sensitivity | Specificity | PPV | NPV |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for name, mdl in p["models"].items():
        h = mdl["hgs_vs_other"]
        L.append(
            f"| {NAMES.get(name, name)} | {h['n_positive']}/{h['n']} | {_ci(h['auroc_ci'])} | {_f(h['auprc'])} ({_f(h['prevalence'])}) | "
            f"{_f(h['sensitivity'])} | {_f(h['specificity'])} | {_f(h['ppv'])} | {_f(h['npv'])} |"
        )
    L += [
        "",
        f"With only {p['models'][p['primary_pathway']]['hgs_vs_other']['n_positive']} positive test images, "
        "possibly from a few patients, these estimates are highly uncertain.",
        "",
        "## Ablations: what changed when a component was removed?",
        "",
        "| Question | Metric | A - B | 95% CI | Resamples with A better |",
        "|---|---|---:|---|---:|",
    ]
    for a in p["ablations"]:
        for metric in ("balanced_accuracy", "macro_auroc", "hgs_auroc"):
            d = a[metric]
            ci = d["ci95"]
            L.append(
                f"| {a['question']} | {metric} | {_f(d['difference'])} | "
                f"{'n/a' if not ci else f'{ci[0]:.3f} to {ci[1]:.3f}'} | {_f(d['fraction_resamples_a_better'], 2)} |"
            )
    seg = p["segmentation"]
    L += [
        "",
        "## Segmentation (automatic localisation)",
        "",
        f"Reference: {seg['reference']}. {seg['failure_counting']}.",
        "",
        "| Subset | n | Candidate rate | Localiser Dice (mean / median) | SAM auto Dice (mean / median) | SAM oracle-prompt Dice (mean / median) |",
        "|---|---:|---:|---|---|---|",
    ]
    for label, block in (
        ("All test images", seg["all_test_images"]),
        ("Lesion categories only", seg["lesion_categories_only"]),
    ):
        L.append(
            f"| {label} | {block['n']} | {_f(block['candidate_rate'])} | "
            + " | ".join(
                f"{_f(block[k]['dice']['mean'])} / {_f(block[k]['dice']['median'])}"
                for k in ("coarse", "sam_auto", "sam_oracle")
            )
            + " |"
        )
    d = seg["sam_minus_coarse_mean_dice"]
    L += [
        "",
        f"SAM minus localiser mean Dice: {d['difference']:.3f} (95% CI {d['ci95'][0]:.3f} to {d['ci95'][1]:.3f}). {seg['oracle_prompt_note']}",
        "",
        "## Sensitivity analyses (primary pathway)",
        "",
    ]
    for group in ("near_duplicate", "colour_content"):
        g = p["sensitivity_analyses"][group]
        L.append(f"- {g['description']}:")
        for v in g.values():
            if isinstance(v, dict):
                L.append(
                    f"  - {v['subset']}: n={v['n']}"
                    + (
                        f" (suppressed: {v['reason']})"
                        if v["suppressed"]
                        else f", balanced acc. {_f(v['balanced_accuracy'])}, accuracy {_f(v['accuracy'])}"
                    )
                )
    fc = p["failure_cases"]
    L += ["", "## Failure cases", "", fc["selection_rule"], ""]
    for key in ("confident_errors", "abstained", "hgs_below_threshold"):
        L.append(
            f"- {key}: "
            + (
                ", ".join(
                    f"{c['image_id']} (true {c['true']}, top {c['highest_scoring']} {c['top_score']:.2f})"
                    for c in fc[key]
                )
                or "none"
            )
        )
    L += ["", "## Provenance", ""]
    for k, v in p["provenance"].items():
        if k != "heads_sha256":
            L.append(f"- {k}: `{v}`")
    L.append("")
    return "\n".join(L)
