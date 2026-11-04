"""Stage 2: DINOv2 patch localiser (development masks only) and box prompts.

* Grouped CV folds are created once here and reused by every later stage.
* Development images receive OUT-OF-FOLD localiser outputs, so ROI features used
  for classifier training match what the automatic system produces at test time.
* Test images receive outputs from the localiser fitted on all development data.
* The detection threshold is chosen by mean Dice on development OOF predictions.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ovascope_ml import paths
from ovascope_ml.data.imaging import load_mask_file
from ovascope_ml.data.splits import group_keys, grouped_cv_folds
from ovascope_ml.models.geometry import Letterbox, dice_iou
from ovascope_ml.models.heads import LinearHead
from ovascope_ml.models.localizer import LocalizerParams, PatchLocalizer, patch_targets, region_from_grid
from ovascope_ml.training.common import FeatureStore, dev_view, load_manifest, log, stage_dir, test_inputs


def _near_dup_clusters() -> dict[str, int]:
    table = json.loads((paths.processed_dir() / "image_table.json").read_text(encoding="utf-8"))
    return {r["image_id"]: r["near_dup_cluster"] for r in table if r["near_dup_cluster"] is not None}


def folds_for(cfg: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    out = stage_dir(cfg, "localizer") / "folds.json"
    if out.exists():
        d = json.loads(out.read_text(encoding="utf-8"))
        return d["image_ids"], np.array(d["folds"])
    m = load_manifest(cfg)
    dv = dev_view(m)
    groups = group_keys(m.by_split("development"), _near_dup_clusters())
    folds = grouped_cv_folds(dv.labels.tolist(), groups, n_folds=cfg["cv"]["n_folds"], seed=cfg["seed"])
    out.write_text(json.dumps({"image_ids": dv.image_ids, "folds": folds.tolist(), "groups": groups}), encoding="utf-8")
    return dv.image_ids, folds


def _fit(x: np.ndarray, y: np.ndarray, c: float) -> LinearHead:
    scaler = StandardScaler().fit(x)
    lr = LogisticRegression(C=c, max_iter=1000)
    lr.fit(scaler.transform(x), y)
    return LinearHead.from_sklearn(scaler, lr, classes=["background", "target_region"], kind="sigmoid")


def run(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    sd = stage_dir(cfg, "localizer")
    done = sd / "localizer_outputs.json"
    if done.exists() and not force:
        log("localizer: cached")
        return json.loads(done.read_text(encoding="utf-8"))
    lc = cfg["localizer"]
    rng = np.random.default_rng(cfg["seed"])
    m = load_manifest(cfg)
    dv, ti = dev_view(m), test_inputs(m)
    root = paths.mmotu_root()
    feats = FeatureStore(stage_dir(cfg, "features"))
    patches_all, pmeta = feats.load("dino_patches", mmap=True)
    index = {iid: i for i, iid in enumerate(pmeta["image_ids"])}
    sizes = feats.rows("image_sizes", pmeta["image_ids"])
    size_of = {iid: (int(w), int(h)) for iid, (w, h) in zip(pmeta["image_ids"], sizes, strict=True)}
    img_size = cfg["backbone"]["img_size"]
    g = patches_all.shape[1]
    ids, folds = folds_for(cfg)
    assert ids == dv.image_ids

    # Development patch targets from reference masks (development only).
    log("localizer: building development patch targets")
    targets = np.zeros((len(dv.image_ids), g, g), dtype=np.float32)
    valid = np.zeros((len(dv.image_ids), g, g), dtype=bool)
    gt_masks: list[np.ndarray] = []
    for k, (iid, mpath) in enumerate(zip(dv.image_ids, dv.mask_paths, strict=True)):
        w, h = size_of[iid]
        lb = Letterbox(w, h, img_size)
        mask = load_mask_file(root / mpath) if mpath else np.zeros((h, w), bool)
        gt_masks.append(mask)
        targets[k] = patch_targets(lb.apply_mask(mask), g)
        valid[k] = patch_targets(lb.valid_region(), g) > 0.5

    def sample(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        per = lc["samples_per_image"]
        for k in rows:
            t, v = targets[k].ravel(), valid[k].ravel()
            pos = np.nonzero((t >= lc["positive_fraction"]) & v)[0]
            neg = np.nonzero((t == 0) & v)[0]
            pos = rng.choice(pos, size=min(per, len(pos)), replace=False) if len(pos) else pos
            neg = rng.choice(neg, size=min(per, len(neg)), replace=False) if len(neg) else neg
            tok = np.asarray(patches_all[index[dv.image_ids[k]]], dtype=np.float32).reshape(g * g, -1)
            xs.append(tok[np.concatenate([pos, neg])])
            ys.append(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]))
        return np.concatenate(xs), np.concatenate(ys)

    base = LocalizerParams(
        threshold=0.5, smooth_sigma=lc["smooth_sigma"], min_component_patches=lc["min_component_patches"]
    )
    oof_grid = np.zeros((len(dv.image_ids), g, g), dtype=np.float32)
    for f in np.unique(folds):
        tr = np.nonzero(folds != f)[0]
        va = np.nonzero(folds == f)[0]
        x, y = sample(tr)
        log(f"localizer: fold {f} fit on {len(y)} patches")
        loc = PatchLocalizer(_fit(x, y, lc["C"]), base)
        for k in va:
            iid = dv.image_ids[k]
            w, h = size_of[iid]
            oof_grid[k] = loc.prob_grid(
                np.asarray(patches_all[index[iid]], dtype=np.float32), Letterbox(w, h, img_size)
            )

    # Threshold by mean development OOF Dice (source resolution).
    best_t, best_dice, table = None, -1.0, []
    for t in lc["thresholds"]:
        p = LocalizerParams(
            threshold=t, smooth_sigma=lc["smooth_sigma"], min_component_patches=lc["min_component_patches"]
        )
        dices, found = [], 0
        for k, iid in enumerate(dv.image_ids):
            w, h = size_of[iid]
            res = region_from_grid(oof_grid[k], Letterbox(w, h, img_size), p)
            pred = res.coarse_mask if res.coarse_mask is not None else np.zeros((h, w), bool)
            found += res.status == "candidate_found"
            dices.append(dice_iou(pred, gt_masks[k])[0])
        md = float(np.mean(dices))
        table.append({"threshold": t, "mean_dev_oof_dice": md, "candidate_rate": found / len(dv.image_ids)})
        log(f"localizer: t={t:.2f} dev OOF dice={md:.3f} found={found}")
        if md > best_dice:
            best_t, best_dice = t, md
    assert best_t is not None
    params = LocalizerParams(
        threshold=float(best_t), smooth_sigma=lc["smooth_sigma"], min_component_patches=lc["min_component_patches"]
    )

    # Final localiser on all development data -> test grids.
    x, y = sample(np.arange(len(dv.image_ids)))
    log(f"localizer: final fit on {len(y)} patches")
    final_head = _fit(x, y, lc["C"])
    final_head.meta = {"trained_on": "development split reference masks", "backbone": pmeta["model_id"]}
    final = PatchLocalizer(final_head, params)
    final_head.save(sd / "localizer_head.npz")
    params.save(sd / "localizer_params.json")
    test_grid = np.zeros((len(ti.image_ids), g, g), dtype=np.float32)
    for k, iid in enumerate(ti.image_ids):
        w, h = size_of[iid]
        test_grid[k] = final.prob_grid(np.asarray(patches_all[index[iid]], dtype=np.float32), Letterbox(w, h, img_size))

    # Boxes for every image with the frozen threshold.
    boxes: dict[str, Any] = {}
    for ids_, grids, source in ((dv.image_ids, oof_grid, "dev_oof"), (ti.image_ids, test_grid, "final")):
        for k, iid in enumerate(ids_):
            w, h = size_of[iid]
            res = final.from_grid(grids[k], Letterbox(w, h, img_size))
            boxes[iid] = {
                "status": res.status,
                "box": res.box,
                "max_score": res.max_score,
                "component_patches": res.component_patches,
                "localizer": source,
            }
    np.save(sd / "grid_dev_oof.npy", oof_grid)
    np.save(sd / "grid_test.npy", test_grid)
    out = {
        "threshold_table": table,
        "selected_threshold": best_t,
        "selected_dev_oof_dice": best_dice,
        "dev_ids": dv.image_ids,
        "test_ids": ti.image_ids,
        "boxes": boxes,
    }
    done.write_text(json.dumps(out), encoding="utf-8")
    log(f"localizer: done (threshold {best_t}, dev OOF dice {best_dice:.3f})")
    return out
