"""Stage 4: DINOv2 features of region-of-interest crops.

ROI sources
  sam     : bounding box of the automatic SAM mask (primary ROI pathway)
  coarse  : localiser box without SAM refinement (ablation: what does SAM add?)
  gt      : reference-mask box. Development: allowed for training an oracle head.
            Test: EVALUATION ONLY (oracle upper bound), stored under eval_only.
Missing ROI -> NaN row; downstream pathways fall back to whole-image features and
report how often that happened.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ovascope_ml import paths
from ovascope_ml.data.imaging import load_mask_file, load_rgb_file
from ovascope_ml.data.validation import DATASET_LIMITS
from ovascope_ml.models.backbones import Dinov2Extractor
from ovascope_ml.models.geometry import expand_box, mask_bbox
from ovascope_ml.training import localizer_stage, sam_stage
from ovascope_ml.training.common import FeatureStore, load_manifest, log, stage_dir


def crop(rgb: np.ndarray, box: tuple[int, int, int, int], margin: float) -> np.ndarray:
    x1, y1, x2, y2 = expand_box(box, margin, rgb.shape[1], rgb.shape[0])
    return rgb[y1:y2, x1:x2]


def run(cfg: dict[str, Any], force: bool = False) -> FeatureStore:
    from ovascope_ml.evaluation.reference import test_reference_masks

    store = FeatureStore(stage_dir(cfg, "roi"))
    eval_store = FeatureStore(stage_dir(cfg, "roi") / "eval_only")
    if (
        not force
        and all(store.has(n) for n in ("roi_sam", "roi_coarse", "roi_gt_dev"))
        and eval_store.has("roi_gt_test")
    ):
        log("roi: cached")
        return store
    loc = localizer_stage.run(cfg)
    sam_dir = sam_stage.run(cfg)
    m = load_manifest(cfg)
    root = paths.mmotu_root()
    margin = cfg["roi"]["margin"]
    dino = Dinov2Extractor(cfg["backbone"]["key"], img_size=cfg["roi"]["img_size"])
    dev = m.by_split("development")
    test = m.by_split("test")
    ref_test = test_reference_masks(m)

    def boxes_for(records, source: str) -> list[tuple[int, int, int, int] | None]:
        out = []
        for r in records:
            if source == "coarse":
                b = loc["boxes"][r.image_id]
                out.append(tuple(b["box"]) if b["status"] == "candidate_found" else None)
            elif source == "sam":
                p = sam_dir / "auto" / f"{r.image_id}.npz"
                out.append(mask_bbox(sam_stage.load_mask(p)[0]) if p.exists() else None)
            elif source == "gt_dev":
                out.append(
                    mask_bbox(load_mask_file(root / r.mask_path_if_available)) if r.mask_path_if_available else None
                )
            elif source == "gt_test":
                mk = ref_test.get(r.image_id)
                out.append(mask_bbox(mk) if mk is not None else None)
        return out

    jobs = [
        ("roi_sam", dev + test, store),
        ("roi_coarse", dev + test, store),
        ("roi_gt_dev", dev, store),
        ("roi_gt_test", test, eval_store),
    ]
    t0 = time.time()
    for name, records, target in jobs:
        if target.has(name) and not force:
            continue
        source = name.removeprefix("roi_")
        boxes = boxes_for(records, source)
        feats = np.full((len(records), 2 * int(dino._load().embed_dim)), np.nan, dtype=np.float32)
        todo = [i for i, b in enumerate(boxes) if b is not None]
        for s in range(0, len(todo), 16):
            idx = todo[s : s + 16]
            crops = []
            for i in idx:
                box = boxes[i]
                assert box is not None
                crops.append(crop(load_rgb_file(root / records[i].image_path, DATASET_LIMITS).rgb, box, margin))
            outs = dino.extract(crops)
            for i, o in zip(idx, outs, strict=True):
                feats[i] = o.global_features()
        target.save(
            name,
            feats,
            [r.image_id for r in records],
            {
                "model_id": dino.model_id,
                "img_size": dino.img_size,
                "margin": margin,
                "n_missing": int(sum(b is None for b in boxes)),
            },
        )
        log(f"roi: {name} done ({len(todo)} crops, {time.time() - t0:.0f}s)")
    return store
