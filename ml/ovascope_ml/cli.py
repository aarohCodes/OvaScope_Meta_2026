"""OvaScope ML command line.

ovascope-ml data-audit                      manifest, validation, shortcut audit
ovascope-ml verify-mirror                   compare official Drive files with the mirror copy
ovascope-ml fetch-models                    download pinned Meta weights (DINOv2, SAM 2.1)
ovascope-ml provider-status                 print provider readiness
ovascope-ml train --config CFG [--stage S]  features -> localizer -> sam -> roi -> classifiers
ovascope-ml evaluate --config CFG           join test labels, compute metrics, write reports
ovascope-ml export-bundle --config CFG      package an evaluated experiment for inference
ovascope-ml report --config CFG             re-render markdown from metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ovascope_ml import paths

STAGES = ("features", "localizer", "sam", "roi", "classifiers")


def _cfg(args):
    from ovascope_ml.training.common import load_config

    return load_config(Path(args.config))


def cmd_data_audit(args) -> int:
    from ovascope_ml.data.audit import run_audit

    root = Path(args.root) if args.root else paths.mmotu_root()
    if not (root / "OTU_2d" / "train_cls.txt").exists():
        print(f"MMOTU OTU_2d not found under {root}. See docs/DATASET_READINESS.md for how to obtain it.")
        return 2
    s = run_audit(root)
    v = s["validation"]
    print(
        json.dumps(
            {
                "n_images": s["n_images"],
                "issue_counts": v["issue_counts"],
                "near_duplicates": v["near_duplicates"],
                "metadata_probe": s["shortcut_audit"]["metadata_probe"]["mmotu8"],
            },
            indent=2,
        )
    )
    errors = [i for i in v["issues"] if i["severity"] == "error"]
    if errors:
        print(f"{len(errors)} error-level issue types found; see docs/generated/DATA_AUDIT.md")
    return 0


def cmd_verify_mirror(args) -> int:
    from ovascope_ml.data.hashing import sha256_file

    official = Path(args.official_dir)
    mirror = paths.mmotu_root()
    files = sorted(p for p in official.rglob("*") if p.is_file())
    if not files:
        print(f"no official files under {official}")
        return 2
    matched, mismatched, missing = 0, [], []
    for p in files:
        rel = p.relative_to(official)
        parts = rel.parts[1:] if rel.parts[0] == "MMOTU" else rel.parts
        q = mirror.joinpath(*parts)
        if not q.exists():
            missing.append(str(Path(*parts)))
        elif sha256_file(p) == sha256_file(q):
            matched += 1
        else:
            mismatched.append(str(Path(*parts)))
    result = {
        "official_files_checked": len(files),
        "identical": matched,
        "different": mismatched,
        "missing_in_mirror": missing,
    }
    print(json.dumps(result, indent=2))
    out = paths.evaluation_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "mirror_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if not mismatched and not missing else 1


def cmd_fetch_models(args) -> int:
    from huggingface_hub import hf_hub_download

    from ovascope_ml.models.registry import REGISTRY

    keys = args.keys or ["dinov2_vits14", "sam2.1_hiera_tiny"]
    for key in keys:
        spec = REGISTRY[key]
        for f in spec.files:
            path = hf_hub_download(spec.repo_id, f, revision=spec.revision, token=False)
            print(f"{spec.repo_id}@{spec.revision[:12]} {f} -> cached ({Path(path).stat().st_size} bytes)")
    print("torchvision ResNet-50 comparator weights download on first `train` (torch hub cache).")
    return 0


def cmd_provider_status(args) -> int:
    from dataclasses import asdict

    from ovascope_ml.inference.pipeline import AnalysisEngine

    eng = AnalysisEngine.from_environment()
    print(
        json.dumps(
            {
                "mode": eng.mode(),
                "providers": [asdict(s) for s in eng.statuses()],
                "missing": eng.missing_requirements(),
            },
            indent=2,
            default=str,
        )
    )
    return 0


def cmd_train(args) -> int:
    from ovascope_ml.training import classifier_stage, features, localizer_stage, roi_stage, sam_stage

    cfg = _cfg(args)
    run = {
        "features": features.run,
        "localizer": localizer_stage.run,
        "sam": sam_stage.run,
        "roi": roi_stage.run,
        "classifiers": classifier_stage.run,
    }
    stages = [args.stage] if args.stage else list(STAGES)
    for s in stages:
        run[s](cfg, force=args.force)
    return 0


def cmd_evaluate(args) -> int:
    from ovascope_ml.evaluation.evaluate import run

    run(_cfg(args))
    print("wrote metrics.json, predictions_test.csv, evaluation_config.json, EVALUATION_REPORT.md")
    return 0


def cmd_export_bundle(args) -> int:
    from ovascope_ml.training.bundle import export

    out = export(_cfg(args))
    print(f"bundle exported: {out.name} (now ACTIVE)")
    return 0


def cmd_report(args) -> int:
    from ovascope_ml.evaluation.report import render_evaluation_markdown
    from ovascope_ml.training.common import experiment_dir

    cfg = _cfg(args)
    payload = json.loads((experiment_dir(cfg) / "metrics.json").read_text(encoding="utf-8"))
    out = paths.REPO_ROOT / "docs" / "generated" / "EVALUATION_REPORT.md"
    out.write_text(render_evaluation_markdown(payload), encoding="utf-8")
    print(f"rendered {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="ovascope-ml", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("data-audit")
    p.add_argument("--root")
    p.set_defaults(fn=cmd_data_audit)
    p = sub.add_parser("verify-mirror")
    p.add_argument("--official-dir", default=str(paths.data_dir() / "raw" / "mmotu_gdrive"))
    p.set_defaults(fn=cmd_verify_mirror)
    p = sub.add_parser("fetch-models")
    p.add_argument("keys", nargs="*")
    p.set_defaults(fn=cmd_fetch_models)
    p = sub.add_parser("provider-status")
    p.set_defaults(fn=cmd_provider_status)
    for name, fn in (
        ("train", cmd_train),
        ("evaluate", cmd_evaluate),
        ("export-bundle", cmd_export_bundle),
        ("report", cmd_report),
    ):
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/mmotu_v1.yaml")
        if name == "train":
            p.add_argument("--stage", choices=STAGES)
            p.add_argument("--force", action="store_true")
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
