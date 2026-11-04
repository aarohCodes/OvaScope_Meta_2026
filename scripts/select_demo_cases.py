"""Select demonstration cases from HELD-OUT predictions by fixed rules (no hand-picking).

Run after `ovascope-ml evaluate`. Writes docs/generated/DEMO_CASES.md (image IDs and
aggregate context only; images stay local under data/raw and are never committed).

Rules (primary pathway, test split):
  A. Correct, difficult: correctly classified, not abstained, NO development near-duplicate,
     smallest margin between the top two scores.
  B. Benign look-alike: a non-"High grade serous" image, correctly classified and not abstained,
     with the highest "High grade serous" score among such images.
  C. Failure handled: the lowest-scoring abstained case; and the highest-scoring confident error.
Selection is curated by these rules and is disclosed as curated in the demo.

Usage: python scripts/select_demo_cases.py [experiment_id]
"""

from __future__ import annotations

import csv
import json
import sys

from ovascope_ml import paths


def main() -> int:
    exp_id = sys.argv[1] if len(sys.argv) > 1 else "mmotu-otu2d-v1"
    exp = paths.experiments_dir() / exp_id
    metrics = json.loads((exp / "metrics.json").read_text(encoding="utf-8"))
    primary = metrics["primary_pathway"]
    table = {
        r["image_id"]: r for r in json.loads((paths.processed_dir() / "image_table.json").read_text(encoding="utf-8"))
    }
    dev_clusters = {r["near_dup_cluster"] for r in table.values() if r["split"] == "development"}
    rows = []
    with (exp / "predictions_test.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["variant"] != primary:
                continue
            scores = sorted((float(v) for k, v in r.items() if k.startswith("score_")), reverse=True)
            r["margin"] = scores[0] - scores[1]
            r["hgs"] = float(r["score_high_grade_serous"])
            r["has_dev_dup"] = table[r["image_id"]]["near_dup_cluster"] in dev_clusters
            r["abstained_b"] = r["abstained"] == "True"
            rows.append(r)
    correct = [r for r in rows if r["predicted"] == r["true_label"] and not r["abstained_b"]]
    a = min((r for r in correct if not r["has_dev_dup"]), key=lambda r: r["margin"], default=None)
    b = max((r for r in correct if r["true_label"] != "high_grade_serous"), key=lambda r: r["hgs"], default=None)
    abst = min((r for r in rows if r["abstained_b"]), key=lambda r: float(r["top_score"]), default=None)
    err = max(
        (r for r in rows if r["predicted"] != r["true_label"] and not r["abstained_b"]),
        key=lambda r: float(r["top_score"]),
        default=None,
    )

    def line(tag: str, rule: str, r) -> str:
        if r is None:
            return f"| {tag} | {rule} | none matched | | | | |"
        stem = r["image_id"].removeprefix("mmotu2d-")
        return (
            f"| {tag} | {rule} | `{r['image_id']}` (`OTU_2d/images/{stem}.JPG`) | {r['true_label']} | "
            f"{r['predicted']} | {float(r['top_score']):.2f} | {r['hgs']:.2f} |"
        )

    m = metrics["models"][primary]
    out = [
        "# Demonstration cases (generated, curated by fixed rules)",
        "",
        "Research prototype. Not for diagnosis, screening, or treatment decisions.",
        "",
        f"Experiment `{exp_id}`, primary pathway `{primary}`, held-out test split. Selection rules are in "
        "`scripts/select_demo_cases.py`. These cases are **curated** and are not representative on their own; "
        f"aggregate context: balanced accuracy {m['mmotu8']['balanced_accuracy']:.3f} "
        f"(95% CI {m['mmotu8']['balanced_accuracy_ci']['ci95'][0]:.3f}-{m['mmotu8']['balanced_accuracy_ci']['ci95'][1]:.3f}), "
        f"image level, n={metrics['scope']['n_test']}. {metrics['scope']['external_validation']}",
        "",
        "| Case | Rule | Image (local path) | Reference category | Highest-scoring | Top score | HGS score |",
        "|---|---|---|---|---|---:|---:|",
        line("A", "correct, no dev near-duplicate, smallest top-2 margin", a),
        line("B", "non-HGS, correct, highest HGS score (look-alike)", b),
        line("C1", "abstained, lowest top score", abst),
        line("C2", "confident error (highest top score among wrong, not abstained)", err),
        "",
        "Images are local research copies and are never committed or redistributed.",
        "",
    ]
    dest = paths.REPO_ROOT / "docs" / "generated" / "DEMO_CASES.md"
    dest.write_text("\n".join(out), encoding="utf-8")
    print(dest.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
