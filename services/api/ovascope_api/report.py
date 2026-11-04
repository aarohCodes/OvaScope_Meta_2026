"""Deterministic research report (template only; no language model).

Contains only verified structured fields from the stored result and the
evaluation artifact. It never states or implies a diagnosis.
"""

from __future__ import annotations

import html
from typing import Any

from ovascope_ml import RESEARCH_NOTICE


def _e(v: Any) -> str:
    return html.escape("n/a" if v is None else str(v))


def build_report(
    study: dict[str, Any],
    result: dict[str, Any] | None,
    evaluation: dict[str, Any] | None,
    annotations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rep: dict[str, Any] = {
        "notice": RESEARCH_NOTICE,
        "study": study,
        "result": result,
        "researcher_annotations": {
            "note": "Manual polygons drawn by a researcher; not model output and not used by the automatic pipeline.",
            "items": annotations or [],
        },
        "evaluation_summary": None,
        "generated_by": "ovascope deterministic template v1",
    }
    if evaluation and evaluation.get("available") and result and result.get("classification"):
        pathway = result["classification"]["pathway"]
        mdl = evaluation["models"].get(pathway, {})
        rep["evaluation_summary"] = {
            "experiment_id": evaluation["experiment_id"],
            "pathway": pathway,
            "scope": evaluation["scope"],
            "balanced_accuracy": mdl.get("mmotu8", {}).get("balanced_accuracy_ci"),
            "macro_auroc": mdl.get("mmotu8", {}).get("macro_auroc_ci"),
            "abstention": mdl.get("abstention"),
        }
    return rep


def render_markdown(rep: dict[str, Any]) -> str:
    r, s = rep["result"], rep["study"]
    L = [
        "# OvaScope research report",
        "",
        f"**{RESEARCH_NOTICE}**",
        "",
        f"- Study: `{s['id']}`  (input SHA-256 `{s['input_sha256']}`)",
        f"- Image: {s['image_format']} {s['width']}x{s['height']} px, uploaded {s['created_at']}",
    ]
    if r is None:
        L += ["", "No analysis result is available for this study."]
        return "\n".join(L) + "\n"
    L += [
        f"- Result status: **{r['status']}**  (schema {r['schema_version']}, created {r['created_at']})",
        f"- Model: {r.get('model_id') or 'none'} {r.get('model_version') or ''}",
        f"- Checkpoint SHA-256: `{r.get('checkpoint_sha256') or 'n/a'}`",
        f"- Calibration status: {r['calibration_status']}",
        "",
    ]
    if r.get("missing_requirements"):
        L += (
            ["## Model unavailable", ""]
            + [f"- {m['component']}: {m['requirement']} ({m['action']})" for m in r["missing_requirements"]]
            + [""]
        )
    if r.get("error"):
        L += ["## Error", "", f"- {r['error']['code']}: {r['error']['message']}", ""]
    seg = r.get("segmentation")
    if seg:
        L += [
            "## Segmentation (separate from classification)",
            "",
            f"- Status: {seg['status']}; automatic: {seg['automatic']}",
            f"- Provider: {seg.get('provider_id')} / {seg.get('model_id')}",
            f"- Prompt: {seg.get('prompt_source')} box {seg.get('prompt_box_xyxy')}",
            f"- SAM predicted IoU (self-estimate, not validated): {seg.get('sam_predicted_iou')}",
            f"- Mask area fraction: {seg.get('mask_area_fraction')}",
            f"- Note: {seg.get('message') or 'A segmentation mask marks a candidate region; it is not a diagnosis.'}",
            "",
        ]
    c = r.get("classification")
    if c:
        L += [
            "## Classifier output",
            "",
            f"- Pathway: {c['pathway']} - {c['pathway_reason']}",
            f"- Highest-scoring MMOTU category: {c['highest_scoring_display']}",
            f"- Abstention: {r['abstention_status']}"
            + (f" - {r['abstention_reason_or_null']}" if r.get("abstention_reason_or_null") else ""),
            f"- Score semantics: {c['score_semantics']}",
            "",
            "| Category | Model score |",
            "|---|---:|",
        ]
        L += [f"| {x['display']} | {x['score']:.3f} |" for x in c["scores"]] + [""]
    L += ["## Quality checks (warnings are heuristics, not clinical adequacy)", ""]
    L += [f"- [{q['status']}] {q['category']}/{q['id']}: {q['message']}" for q in r.get("quality_checks", [])]
    if r.get("warnings"):
        L += ["", "## Warnings", ""] + [f"- {w}" for w in r["warnings"]]
    ev = rep.get("evaluation_summary")
    L += ["", "## Evaluation context", ""]
    if ev:
        ba, au = ev["balanced_accuracy"], ev["macro_auroc"]
        L += [
            f"- Experiment {ev['experiment_id']}, pathway {ev['pathway']}, {ev['scope']['evaluation_unit']}-level, "
            f"n={ev['scope']['n_test']} held-out images.",
            f"- Balanced accuracy {ba['point']:.3f} (95% CI {ba['ci95'][0]:.3f}-{ba['ci95'][1]:.3f}); "
            f"macro AUROC {au['point']:.3f} (95% CI {au['ci95'][0]:.3f}-{au['ci95'][1]:.3f}).",
            f"- {ev['scope']['statement']}",
            f"- {ev['scope']['external_validation']}",
            f"- {ev['scope']['ci_note']}",
        ]
    else:
        L += ["- No evaluation artifact applies to this result."]
    L += ["", "## Processing times (ms)", ""] + [f"- {k}: {v}" for k, v in r.get("processing_times", {}).items()]
    ann = rep.get("researcher_annotations", {}).get("items", [])
    L += ["", "## Researcher annotations (manual, not model output)", ""]
    L += [
        f"- {a['id'][:8]}: {len(a['polygon'])}-point polygon" + (f" - {a['note']}" if a.get("note") else "")
        for a in ann
    ] or ["- None."]
    L += ["", "---", "Generated by a deterministic template. Not for diagnosis, screening, or treatment decisions.", ""]
    return "\n".join(L)


def render_html(rep: dict[str, Any]) -> str:
    md = render_markdown(rep)
    body = []
    in_table = False
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-:"):
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
            body.append("<tr>" + "".join(f"<td>{_e(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        if line.startswith("# "):
            body.append(f"<h1>{_e(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{_e(line[3:])}</h2>")
        elif line.startswith("- "):
            body.append(f"<p class='li'>{_e(line[2:]).replace('**', '')}</p>")
        elif line.strip() == "---":
            body.append("<hr>")
        elif line.strip():
            body.append(f"<p>{_e(line).replace('**', '')}</p>")
    if in_table:
        body.append("</table>")
    css = (
        "body{font:14px/1.5 system-ui,sans-serif;max-width:860px;margin:32px auto;padding:0 16px;color:#0f172a}"
        "h1{font-size:22px}h2{font-size:16px;margin-top:24px;border-bottom:1px solid #cbd5e1}"
        "table{border-collapse:collapse}td{border:1px solid #cbd5e1;padding:4px 8px}"
        ".li{margin:2px 0 2px 12px}.notice{background:#fef3c7;border:1px solid #f59e0b;padding:8px 12px}"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>OvaScope research report</title>"
        f"<style>{css}</style></head><body><p class='notice'>{_e(RESEARCH_NOTICE)}</p>{''.join(body)}</body></html>"
    )
