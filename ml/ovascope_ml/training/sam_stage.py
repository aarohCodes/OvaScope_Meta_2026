"""Stage 3: SAM 2.1 mask refinement from automatic localiser boxes.

For every image with a localiser candidate, SAM is prompted with that box. For test
images only, SAM is additionally prompted with the reference-mask box (oracle prompt,
EVALUATION ONLY, written to eval_only/). The SAM image embedding is computed once per
image and reused for both prompts. Resumable: one small file per image.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ovascope_ml import paths
from ovascope_ml.data.imaging import load_rgb_file
from ovascope_ml.data.validation import DATASET_LIMITS
from ovascope_ml.models.geometry import mask_bbox
from ovascope_ml.providers.sam import Sam2BoxRefiner
from ovascope_ml.training import localizer_stage
from ovascope_ml.training.common import load_manifest, log, stage_dir


def save_mask(path: Path, mask: np.ndarray, **meta: Any) -> None:
    np.savez_compressed(
        path, bits=np.packbits(mask.ravel()), shape=np.array(mask.shape), meta=np.array(json.dumps(meta))
    )


def load_mask(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as z:
        shape = tuple(int(v) for v in z["shape"])
        mask = np.unpackbits(z["bits"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
        return mask, json.loads(str(z["meta"]))


def run(cfg: dict[str, Any], force: bool = False) -> Path:
    from ovascope_ml.evaluation.reference import test_reference_masks

    sd = stage_dir(cfg, "sam")
    auto_dir = sd / "auto"
    oracle_dir = sd / "eval_only" / "oracle_prompt"
    auto_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    summary_path = sd / "sam_summary.json"
    if summary_path.exists() and not force:
        log("sam: cached")
        return sd

    loc = localizer_stage.run(cfg)
    m = load_manifest(cfg)
    root = paths.mmotu_root()
    refiner = Sam2BoxRefiner(cfg["sam"]["key"], multimask_output=cfg["sam"]["multimask_output"])
    st = refiner.status()
    if not st.available:
        raise RuntimeError(f"SAM unavailable: {st.reason_unavailable} -> {st.action_required}")
    test_ids = set(loc["test_ids"])
    ref = test_reference_masks(m)  # eval-only use below
    records = m.by_split("development") + m.by_split("test")
    t0, processed = time.time(), 0
    summary: dict[str, Any] = {}
    for k, rec in enumerate(records):
        iid = rec.image_id
        auto_path = auto_dir / f"{iid}.npz"
        oracle_path = oracle_dir / f"{iid}.npz"
        b = loc["boxes"][iid]
        need_auto = b["status"] == "candidate_found" and not auto_path.exists()
        gt_box = None
        if iid in test_ids and ref.get(iid) is not None:
            gt_box = mask_bbox(ref[iid])  # type: ignore[arg-type]
        need_oracle = gt_box is not None and not oracle_path.exists()
        if need_auto or need_oracle:
            rgb = load_rgb_file(root / rec.image_path, DATASET_LIMITS).rgb
            emb = refiner.embed(rgb)
            prompts: list[tuple[str, tuple[int, int, int, int]]] = []
            if need_auto:
                x1, y1, x2, y2 = (int(v) for v in b["box"])
                prompts.append(("auto", (x1, y1, x2, y2)))
            if need_oracle and gt_box is not None:
                prompts.append(("oracle", gt_box))
            outs = refiner.refine(rgb, [p[1] for p in prompts], embedding=emb)
            for (kind, box), out in zip(prompts, outs, strict=True):
                save_mask(
                    auto_path if kind == "auto" else oracle_path,
                    out.mask,
                    box=list(box),
                    predicted_iou=out.predicted_iou,
                    model_id=refiner.model_id,
                )
            processed += 1
            if processed % 25 == 0:
                rate = (time.time() - t0) / processed
                log(f"sam: {k + 1}/{len(records)} images ({rate:.2f}s/img)")
        if auto_path.exists():
            mask, meta = load_mask(auto_path)
            summary[iid] = {
                "status": "succeeded" if mask.any() else "empty_mask",
                "predicted_iou": meta["predicted_iou"],
                "area_fraction": float(mask.mean()),
            }
        else:
            summary[iid] = {"status": "no_candidate_region", "predicted_iou": None, "area_fraction": None}
    summary_path.write_text(json.dumps({"model_id": refiner.model_id, "images": summary}), encoding="utf-8")
    log(f"sam: done ({processed} images embedded, {time.time() - t0:.0f}s)")
    return sd
