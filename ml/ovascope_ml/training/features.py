"""Stage 1: frozen whole-image features for every image (inputs only; no labels, no masks)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ovascope_ml import paths
from ovascope_ml.data.imaging import load_rgb_file
from ovascope_ml.data.validation import DATASET_LIMITS
from ovascope_ml.models.backbones import Dinov2Extractor, HandcraftedExtractor, ResNet50Extractor
from ovascope_ml.training.common import FeatureStore, load_manifest, log, stage_dir

BATCH = 8


def run(cfg: dict[str, Any], force: bool = False) -> FeatureStore:
    store = FeatureStore(stage_dir(cfg, "features"))
    names = ("dino_global", "dino_patches", "resnet50", "handcrafted", "image_sizes")
    if not force and all(store.has(n) for n in names):
        log("features: cached")
        return store
    manifest = load_manifest(cfg)
    root = paths.mmotu_root()
    # Inputs only: image paths and ids. Labels/masks are not touched in this stage.
    records = manifest.by_split("development") + manifest.by_split("test")
    ids = [r.image_id for r in records]
    rel_paths = [r.image_path for r in records]
    n = len(ids)

    dino = Dinov2Extractor(cfg["backbone"]["key"], img_size=cfg["backbone"]["img_size"])
    resnet = ResNet50Extractor(img_size=cfg["backbone"]["img_size"])
    hand = HandcraftedExtractor()
    g = dino.grid

    patches_path = store.dir / "dino_patches.npy"
    embed_dim = int(dino._load().embed_dim)
    patches = np.lib.format.open_memmap(patches_path, mode="w+", dtype=np.float16, shape=(n, g, g, embed_dim))
    dino_global = None
    resnet_feats = np.zeros((n, 2048), dtype=np.float32)
    hand_feats = None
    sizes = np.zeros((n, 2), dtype=np.int32)

    t0 = time.time()
    for start in range(0, n, BATCH):
        batch_paths = rel_paths[start : start + BATCH]
        imgs = [load_rgb_file(root / p, DATASET_LIMITS).rgb for p in batch_paths]
        outs = dino.extract(imgs)
        gl = np.stack([o.global_features() for o in outs])
        if dino_global is None:
            dino_global = np.zeros((n, gl.shape[1]), dtype=np.float32)
        dino_global[start : start + len(imgs)] = gl
        patches[start : start + len(imgs)] = np.stack([o.patches for o in outs])
        resnet_feats[start : start + len(imgs)] = resnet.extract(imgs)
        hf = hand.extract(imgs)
        if hand_feats is None:
            hand_feats = np.zeros((n, hf.shape[1]), dtype=np.float32)
        hand_feats[start : start + len(imgs)] = hf
        sizes[start : start + len(imgs)] = [(im.shape[1], im.shape[0]) for im in imgs]
        if (start // BATCH) % 10 == 0:
            done = start + len(imgs)
            log(f"features: {done}/{n} ({(time.time() - t0) / done:.2f}s/img)")
    patches.flush()
    del patches

    assert dino_global is not None and hand_feats is not None
    prov = {"manifest_sha256": manifest.content_hash()}
    store.save(
        "dino_global",
        dino_global,
        ids,
        {**prov, "model_id": dino.model_id, "img_size": dino.img_size, "layout": "concat[CLS, mean(patch tokens)]"},
    )
    # dino_patches already written via memmap; write its sidecar.
    import json

    (store.dir / "dino_patches.json").write_text(
        json.dumps(
            {
                "image_ids": ids,
                "shape": [n, g, g, -1],
                "dtype": "float16",
                **prov,
                "model_id": dino.model_id,
                "img_size": dino.img_size,
            }
        ),
        encoding="utf-8",
    )
    store.save("resnet50", resnet_feats, ids, {**prov, "model_id": resnet.model_id, "img_size": resnet.img_size})
    store.save("handcrafted", hand_feats, ids, {**prov, "model_id": hand.model_id})
    store.save("image_sizes", sizes, ids, {**prov, "layout": "(width, height)"})
    log(f"features: done in {time.time() - t0:.0f}s")
    return store
